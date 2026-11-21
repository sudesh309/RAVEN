"""
Supervisor: plans which agents run, runs them, and synthesises one report.

The plan is deterministic (no LLM decides what gets checked):

* the **quality team** always runs;
* the **cascade agent** runs when the set carries hierarchy information
  (parent / derived-from links or levels);
* the **architecture team** runs only when an architecture model is supplied.

Synthesis merges every agent's findings into a per-requirement scorecard
(``blocked`` / ``needs work`` / ``ready``) plus a per-element list for the
architecture side, and renders it as a dict, Markdown or CSV.
"""

from __future__ import annotations

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .architecture_agents import (AllocationAgent, ArchitectureJustificationAgent,
                                  DerivedRequirementsAgent)
from .base import SEVERITIES, ReviewContext, normalise_requirements
from .cascade import CascadeAgent
from .quality_agents import (CompletenessAgent, ConsistencyAgent, RationaleAgent,
                             WritingQualityAgent)

AGENT_CLASSES = (WritingQualityAgent, CompletenessAgent, RationaleAgent, ConsistencyAgent,
                 CascadeAgent, AllocationAgent, ArchitectureJustificationAgent,
                 DerivedRequirementsAgent)
TEAMS = {"quality": [c.name for c in AGENT_CLASSES if c.domain == "quality"],
         "cascade": [c.name for c in AGENT_CLASSES if c.domain == "cascade"],
         "architecture": [c.name for c in AGENT_CLASSES if c.domain == "architecture"]}


def default_agents(llm=None, names=None) -> list:
    wanted = _expand(names)
    return [cls(llm=llm) for cls in AGENT_CLASSES if wanted is None or cls.name in wanted]


def _expand(names) -> Optional[set]:
    """Agent names, team names ("quality") or "all" → a set of agent names."""
    if not names:
        return None
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out = set()
    for n in names:
        if n == "all":
            return None
        out |= set(TEAMS.get(n, [n]))
    unknown = out - {c.name for c in AGENT_CLASSES}
    if unknown:
        raise ValueError(f"unknown agent(s): {', '.join(sorted(unknown))}; choose from "
                         f"{', '.join(c.name for c in AGENT_CLASSES)} or a team "
                         f"({', '.join(TEAMS)})")
    return out


def plan(state: dict, agents) -> list:
    """Names of the agents that apply to this state, in run order."""
    ctx = ReviewContext.from_state(state)
    return [a.name for a in agents if a.applicable(ctx)[0]]


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def synthesise(state: dict) -> dict:
    """Merge agent results into the review report (pure function of state)."""
    reqs = normalise_requirements(state.get("requirements") or [])
    results = state.get("results") or {}
    findings = [f for r in results.values() for f in r.get("findings", [])]
    per_req = {r["id"]: {"id": r["id"], "text": r["text"],
                         "counts": {s: 0 for s in SEVERITIES}, "rules": [], "status": "ready"}
               for r in reqs}
    per_el: dict = {}
    for f in findings:
        sid = f.get("subject_id") or ""
        if f.get("subject_kind") == "element":
            e = per_el.setdefault(sid, {"element": sid, "counts": {s: 0 for s in SEVERITIES},
                                        "rules": []})
            e["counts"][f["severity"]] += 1
            e["rules"].append(f["rule"])
        elif sid in per_req:
            row = per_req[sid]
            row["counts"][f["severity"]] += 1
            row["rules"].append(f["rule"])
    for row in per_req.values():
        c = row["counts"]
        row["status"] = "blocked" if c["blocker"] else "needs work" if c["major"] else "ready"
        row["rules"] = sorted(set(row["rules"]))
    for e in per_el.values():
        e["rules"] = sorted(set(e["rules"]))
    n = len(per_req)
    statuses = [r["status"] for r in per_req.values()]
    totals = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
    teams = {}
    for name, r in results.items():
        t = teams.setdefault(r.get("domain", "quality"), {"agents": [], "counts": {s: 0 for s in SEVERITIES}})
        t["agents"].append(name)
        for s in SEVERITIES:
            t["counts"][s] += r.get("counts", {}).get(s, 0)
    verdict = ("not ready — blockers open" if totals["blocker"]
               else "needs work" if totals["major"] else "ready for review")
    return {
        "summary": {
            "requirements": n,
            "architecture": bool(state.get("architecture")),
            "agents_run": [k for k, v in results.items() if v.get("status") == "ok"],
            "agents_skipped": {k: v.get("reason", "") for k, v in results.items()
                               if v.get("status") == "skipped"},
            "agents_failed": {k: v.get("reason", "") for k, v in results.items()
                              if v.get("status") == "error"},
            "findings": totals,
            "candidates": sum(1 for f in findings if f.get("status") == "candidate"),
            "ready": statuses.count("ready"), "needs_work": statuses.count("needs work"),
            "blocked": statuses.count("blocked"),
            "pct_ready": round(100.0 * statuses.count("ready") / n, 1) if n else 0.0,
            "verdict": verdict,
            "llm_calls": sum(r.get("llm_calls", 0) for r in results.values()),
        },
        "teams": teams,
        "metrics": {k: v.get("metrics", {}) for k, v in results.items()},
        "requirements": list(per_req.values()),
        "elements": sorted(per_el.values(), key=lambda e: (-e["counts"]["major"], e["element"])),
        "findings": sorted(findings, key=lambda f: (SEVERITIES.index(f["severity"]),
                                                     f.get("agent", ""), f.get("subject_id", ""))),
        "agents": {k: {kk: vv for kk, vv in v.items() if kk not in ("findings",)}
                   for k, v in results.items()},
    }


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

class ReviewOrchestrator:
    """Run the review agents over a requirement set (and optional model).

    ``llm`` is an :class:`~reqgraph.llm.LLMProvider` (Ollama, Vertex AI, or any
    object with ``complete(prompt, system=..., temperature=..., json_mode=...)``);
    when given, LLM-capable agents add *candidate* findings.
    """

    def __init__(self, agents=None, *, llm=None, config: Optional[dict] = None,
                 parallel: bool = True):
        self.agents = list(agents) if agents is not None else default_agents(llm)
        for a in self.agents:
            if a.llm is None:
                a.llm = llm
        self.llm = llm
        self.config = dict(config or {})
        self.parallel = parallel

    def describe(self) -> list:
        return [a.describe() for a in self.agents]

    def initial_state(self, requirements, architecture=None, config=None) -> dict:
        arch = architecture
        if arch is not None and not isinstance(arch, dict):
            arch = arch.to_dict()
        return {"requirements": normalise_requirements(requirements),
                "architecture": arch, "config": {**self.config, **(config or {})},
                "plan": [], "results": {}}

    def run(self, requirements, architecture=None, config=None) -> "ReviewReport":
        t0 = time.perf_counter()
        state = self.initial_state(requirements, architecture, config)
        ctx = ReviewContext.from_state(state, llm=self.llm)
        state["plan"] = [a.name for a in self.agents if a.applicable(ctx)[0]]

        def one(agent):
            # each agent gets its own context so caches never race between threads
            return agent.run(ReviewContext.from_state(state, llm=agent.llm)).to_dict()

        if self.parallel and len(self.agents) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(self.agents))) as pool:
                outs = list(pool.map(one, self.agents))
        else:
            outs = [one(a) for a in self.agents]
        state["results"] = {a.name: o for a, o in zip(self.agents, outs)}
        report = synthesise(state)
        report["summary"]["plan"] = state["plan"]
        report["summary"]["duration_s"] = round(time.perf_counter() - t0, 3)
        return ReviewReport(report)


class ReviewReport(dict):
    """The synthesised report (a dict) with Markdown / CSV renderers."""

    def to_dict(self) -> dict:
        return dict(self)

    def to_markdown(self, max_findings: int = 200) -> str:
        s = self["summary"]
        f = s["findings"]
        lines = ["# RAVEN agentic requirement review", "",
                 f"**Verdict:** {s['verdict']}  ",
                 f"**Requirements:** {s['requirements']} — {s['ready']} ready, "
                 f"{s['needs_work']} need work, {s['blocked']} blocked ({s['pct_ready']}% ready)  ",
                 f"**Architecture model:** {'yes' if s['architecture'] else 'not supplied'}  ",
                 f"**Findings:** {f['blocker']} blocker · {f['major']} major · {f['minor']} minor · "
                 f"{f['info']} info ({s['candidates']} candidate, need engineer confirmation)", "",
                 "## Agents", "", "| Agent | Team | Status | Blocker | Major | Minor | Key metrics |",
                 "|---|---|---|---|---|---|---|"]
        for name, a in self["agents"].items():
            c = a.get("counts", {})
            metrics = ", ".join(f"{k}={v}" for k, v in a.get("metrics", {}).items()
                                if not isinstance(v, (dict, list)))[:120]
            status = a["status"] + (f" ({a['reason']})" if a.get("reason") else "")
            lines.append(f"| {name} | {a['domain']} | {status} | {c.get('blocker', 0)} | "
                         f"{c.get('major', 0)} | {c.get('minor', 0)} | {metrics} |")
        lines += ["", "## Requirements needing attention", "",
                  "| ID | Status | Blocker | Major | Rules |", "|---|---|---|---|---|"]
        for r in self["requirements"]:
            if r["status"] != "ready":
                lines.append(f"| {r['id']} | {r['status']} | {r['counts']['blocker']} | "
                             f"{r['counts']['major']} | {', '.join(r['rules'][:6])} |")
        if self["elements"]:
            lines += ["", "## Architecture elements", "", "| Element | Major | Minor | Rules |",
                      "|---|---|---|---|"]
            for e in self["elements"]:
                lines.append(f"| {e['element']} | {e['counts']['major']} | {e['counts']['minor']} | "
                             f"{', '.join(e['rules'])} |")
        lines += ["", "## Findings", "", "| Severity | Agent | Subject | Finding | Hint |",
                  "|---|---|---|---|---|"]
        for x in self["findings"][:max_findings]:
            tag = " *(candidate)*" if x.get("status") == "candidate" else ""
            esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {x['severity']} | {x['agent']} | {esc(x.get('subject_id') or '(set)')} | "
                         f"{esc(x['message'])}{tag} | {esc(x.get('hint', ''))} |")
        if len(self["findings"]) > max_findings:
            lines.append(f"\n… {len(self['findings']) - max_findings} more findings in the JSON report.")
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["severity", "agent", "rule", "subject_kind", "subject_id", "message",
                    "hint", "source", "status"])
        for x in self["findings"]:
            w.writerow([x["severity"], x["agent"], x["rule"], x.get("subject_kind", ""),
                        x.get("subject_id", ""), x["message"], x.get("hint", ""),
                        x.get("source", ""), x.get("status", "")])
        return buf.getvalue()


def review(requirements, architecture=None, *, agents=None, llm=None, config=None) -> ReviewReport:
    """One-call API: ``review(items, load_architecture("model.sysml"))``."""
    return ReviewOrchestrator(default_agents(llm, agents), llm=llm, config=config).run(
        requirements, architecture)
