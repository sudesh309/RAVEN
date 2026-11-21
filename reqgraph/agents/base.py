"""
Agent framework shared by every review agent.

Design rules
------------
* **State in, partial update out.** An agent is a callable
  ``agent(state) -> {"results": {agent.name: result}}`` — exactly the node
  signature LangGraph (and most other graph runtimes) expect, so the same
  object runs under :class:`~reqgraph.agents.orchestrator.ReviewOrchestrator`,
  a LangGraph ``StateGraph``, or a plain loop.
* **State is JSON.** Requirements are ``{"id", "text", "meta"}`` dicts and the
  architecture is :meth:`ArchitectureModel.to_dict` output, so state can be
  checkpointed, sent over MCP, or logged.
* **Deterministic first, LLM second.** ``analyze()`` is rule-based and
  reproducible; it produces *confirmed* findings. An optional LLM pass
  (``llm_instruction``) adds findings marked ``source="llm"`` and
  ``status="candidate"`` — an engineer must accept them. The LLM never removes
  or overrides a deterministic finding.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SEVERITIES = ("blocker", "major", "minor", "info")
_SEV_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Finding:
    agent: str
    rule: str                     # stable machine key, e.g. "missing_rationale"
    severity: str                 # blocker | major | minor | info
    message: str
    subject_id: str = ""          # requirement id or architecture element name ("" = set level)
    subject_kind: str = "requirement"   # requirement | element | set
    hint: str = ""
    evidence: dict = field(default_factory=dict)
    source: str = "deterministic"       # deterministic | llm
    status: str = "confirmed"           # confirmed | candidate

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    agent: str
    domain: str                   # quality | cascade | architecture
    status: str = "ok"            # ok | skipped | error
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)   # [Finding]
    data: dict = field(default_factory=dict)
    duration_s: float = 0.0
    llm_calls: int = 0

    def add(self, rule, severity, message, subject_id="", *, subject_kind=None,
            hint="", evidence=None, source="deterministic", status=None) -> Finding:
        if severity not in _SEV_ORDER:
            severity = "minor"
        f = Finding(agent=self.agent, rule=rule, severity=severity, message=message,
                    subject_id=str(subject_id or ""),
                    subject_kind=subject_kind or ("requirement" if subject_id else "set"),
                    hint=hint, evidence=evidence or {}, source=source,
                    status=status or ("candidate" if source == "llm" else "confirmed"))
        self.findings.append(f)
        return f

    def counts(self) -> dict:
        out = {s: 0 for s in SEVERITIES}
        for f in self.findings:
            out[f.severity] += 1
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() if isinstance(f, Finding) else f for f in self.findings]
        d["counts"] = self.counts()
        return d


# ---------------------------------------------------------------------------
# requirement normalisation and metadata access
# ---------------------------------------------------------------------------

def normalise_requirements(items) -> list:
    """Accept ``(id, text[, meta])`` tuples, ``{"id","text",...}`` dicts or
    bare strings and return ``[{"id", "text", "meta"}]``."""
    out = []
    for i, it in enumerate(items or []):
        if isinstance(it, str):
            rid, text, meta = f"R{i + 1}", it, {}
        elif isinstance(it, dict):
            d = dict(it)
            rid = d.pop("id", None) or d.pop("req_id", None) or f"R{i + 1}"
            text = d.pop("text", None) or d.pop("requirement", None) or ""
            meta = dict(d.pop("meta", None) or {})
            meta.update({k: v for k, v in d.items() if v not in (None, "")})
        else:
            rid = it[0] if len(it) > 0 else f"R{i + 1}"
            text = it[1] if len(it) > 1 else ""
            meta = dict(it[2]) if len(it) > 2 and isinstance(it[2], dict) else {}
        out.append({"id": str(rid if rid not in (None, "") else f"R{i + 1}"),
                    "text": str(text or "").strip(),
                    "meta": {str(k): ("" if v is None else str(v)) for k, v in meta.items()}})
    return out


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def meta_get(meta: dict, *aliases) -> str:
    """First non-empty metadata value among ``aliases`` (case/punctuation-blind)."""
    norm = {_norm_key(k): v for k, v in (meta or {}).items()}
    for a in aliases:
        v = norm.get(_norm_key(a))
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


RATIONALE_KEYS = ("rationale", "justification", "reason", "why", "basis",
                  "rationale_text", "design_rationale")
SOURCE_KEYS = ("source", "origin", "stakeholder", "originator", "reference",
               "source_document", "regulation", "standard")
VERIFICATION_KEYS = ("verification_method", "verification", "verified_by",
                     "method", "vm", "v_method", "test_method", "ivv_method")
LEVEL_KEYS = ("level", "requirement_level", "req_level", "tier", "hierarchy_level")
ALLOCATION_KEYS = ("satisfies", "satisfy", "satisfied_by", "allocated_to",
                   "allocation", "implements", "component", "subsystem", "element")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

class ReviewContext:
    """Per-run view of the state with lazily-built, cached derived data."""

    def __init__(self, requirements, architecture=None, config=None, llm=None):
        self.requirements = normalise_requirements(requirements)
        self.by_id = {r["id"]: r for r in self.requirements}
        self.config = dict(config or {})
        self.llm = llm
        self._arch = architecture
        self._arch_model = None
        self._graphs: dict = {}
        self._cache: dict = {}

    @classmethod
    def from_state(cls, state: dict, llm=None) -> "ReviewContext":
        return cls(state.get("requirements") or [], state.get("architecture"),
                   state.get("config") or {}, llm=llm)

    # -- architecture ------------------------------------------------------
    @property
    def architecture(self):
        if self._arch_model is None and self._arch:
            from ..architecture import ArchitectureModel
            a = self._arch
            self._arch_model = a if isinstance(a, ArchitectureModel) else ArchitectureModel.from_dict(a)
        return self._arch_model

    @property
    def has_architecture(self) -> bool:
        return bool(self._arch)

    # -- parsed requirements ------------------------------------------------
    def graph(self, rid: str):
        """Parsed + enriched :class:`RequirementGraph` for a requirement."""
        if rid not in self._graphs:
            from ..parser import RequirementParser
            from ..quality import enrich
            g = RequirementParser().parse(self.by_id[rid]["text"])
            try:
                enrich(g)
            except Exception:       # analysis is best-effort; parsing is the contract
                pass
            self._graphs[rid] = g
        return self._graphs[rid]

    def role_text(self, rid: str, role_name: str) -> str:
        from ..core import Role
        role = getattr(Role, role_name)
        try:
            return " ".join(n.text for n in self.graph(rid).by_role(role))
        except Exception:
            return ""

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # -- LLM ---------------------------------------------------------------
    def ask_llm_json(self, prompt: str, system: str) -> Optional[dict]:
        if self.llm is None:
            return None
        from ..llm import extract_json
        reply = self.llm.complete(prompt, system=system, temperature=0.1, json_mode=True)
        return extract_json(reply)


# ---------------------------------------------------------------------------
# agent base class
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a senior requirements engineer reviewing a requirement set "
    "(INCOSE Guide to Writing Requirements, ISO/IEC/IEEE 29148, ARP4754B). "
    "Be specific and conservative: report only real problems you can justify "
    "from the material given. Reply with JSON only: "
    '{"findings": [{"subject_id": "...", "rule": "short_snake_case", '
    '"severity": "major|minor|info", "message": "...", "hint": "..."}]}. '
    'Return {"findings": []} when there is nothing to add.')


class ReviewAgent:
    """Base class: subclasses set the class attributes and implement ``analyze``."""

    name = "base"
    domain = "quality"
    title = ""
    description = ""
    requires_architecture = False
    #: when set, the agent can ask an LLM for additional candidate findings
    llm_instruction: str = ""
    llm_batch_size = 12

    def __init__(self, llm=None, *, use_llm: Optional[bool] = None, llm_max_calls: int = 4):
        self.llm = llm
        self.use_llm = use_llm
        self.llm_max_calls = llm_max_calls

    # -- to override ---------------------------------------------------------
    def applicable(self, ctx: ReviewContext) -> tuple:
        """(bool, reason). Default: needs requirements (and a model if declared)."""
        if not ctx.requirements:
            return False, "no requirements supplied"
        if self.requires_architecture and not ctx.has_architecture:
            return False, "no architecture model supplied"
        return True, ""

    def analyze(self, ctx: ReviewContext, result: AgentResult) -> None:
        raise NotImplementedError

    def llm_payload(self, ctx: ReviewContext, result: AgentResult) -> list:
        """Items (JSON-able dicts) the LLM should review. Default: every
        requirement with the deterministic findings already raised on it."""
        by_subject: dict = {}
        for f in result.findings:
            by_subject.setdefault(f.subject_id, []).append(f.rule)
        return [{"id": r["id"], "text": r["text"],
                 "rationale": meta_get(r["meta"], *RATIONALE_KEYS),
                 "already_found": by_subject.get(r["id"], [])}
                for r in ctx.requirements]

    # -- runtime ------------------------------------------------------------
    def describe(self) -> dict:
        return {"name": self.name, "domain": self.domain, "title": self.title or self.name,
                "description": self.description,
                "requires_architecture": self.requires_architecture,
                "llm_capable": bool(self.llm_instruction)}

    def run(self, ctx: ReviewContext) -> AgentResult:
        result = AgentResult(agent=self.name, domain=self.domain)
        t0 = time.perf_counter()
        ok, reason = self.applicable(ctx)
        if not ok:
            result.status, result.reason = "skipped", reason
            return result
        try:
            self.analyze(ctx, result)
        except Exception as exc:              # an agent failure must not sink the review
            result.status, result.reason = "error", f"{type(exc).__name__}: {exc}"
        llm = self.llm if self.llm is not None else ctx.llm
        want_llm = self.use_llm if self.use_llm is not None else \
            bool(ctx.config.get("use_llm", llm is not None))
        if result.status == "ok" and llm is not None and want_llm and self.llm_instruction:
            self._llm_pass(ctx, result, llm)
        result.findings.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 9), f.subject_id, f.rule))
        result.duration_s = round(time.perf_counter() - t0, 3)
        return result

    def _llm_pass(self, ctx, result, llm) -> None:
        items = self.llm_payload(ctx, result)
        if not items:
            return
        cap = int(ctx.config.get("llm_max_calls", self.llm_max_calls))
        known = {r["id"] for r in ctx.requirements}
        if ctx.has_architecture and ctx.architecture is not None:
            known |= {e.name for e in ctx.architecture.elements}
        errors = []
        from ..llm import LLMError, extract_json
        for start in range(0, len(items), self.llm_batch_size):
            if result.llm_calls >= cap:
                result.data["llm_truncated"] = True
                break
            batch = items[start:start + self.llm_batch_size]
            prompt = (f"Task: {self.llm_instruction}\n\nItems:\n"
                      + json.dumps(batch, ensure_ascii=False, indent=1))
            result.llm_calls += 1
            try:
                reply = extract_json(llm.complete(prompt, system=_LLM_SYSTEM,
                                                  temperature=0.1, json_mode=True))
            except (LLMError, ValueError) as exc:
                errors.append(str(exc))
                break                         # a down provider fails every batch alike
            for f in (reply or {}).get("findings", []) or []:
                if not isinstance(f, dict) or not f.get("message"):
                    continue
                sid = str(f.get("subject_id") or "")
                if sid and sid not in known:
                    continue                  # hallucinated subject: drop it
                sev = str(f.get("severity", "minor")).lower()
                result.add(_norm_key(f.get("rule") or "llm_observation") or "llm_observation",
                           "minor" if sev == "blocker" else sev,   # an LLM cannot block a baseline
                           str(f["message"])[:600], sid,
                           subject_kind="requirement" if sid in ctx.by_id else
                           ("element" if sid else "set"),
                           hint=str(f.get("hint") or "")[:600], source="llm")
        if errors:
            result.data["llm_errors"] = errors
        result.data["llm_provider"] = getattr(llm, "describe", lambda: str(llm))()

    def __call__(self, state: dict) -> dict:
        """LangGraph node: read the shared state, return a partial update."""
        ctx = ReviewContext.from_state(state, llm=self.llm)
        return {"results": {self.name: self.run(ctx).to_dict()}}


# ---------------------------------------------------------------------------
# small text helpers shared by agents
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a an the and or of to in on at by for with from into onto within without under
over per as is are be been being shall should must will may can not no any all
each every its it this that these those which who whom whose than then when
while if unless where provided such so least most more less only also both
either neither including include includes during after before until system
""".split())

_WORD_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def content_terms(text: str, extra_stop=()) -> set:
    stop = STOPWORDS | set(extra_stop)
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in stop}


def stem(w: str) -> str:
    """Cheap stemmer: enough to make navigate/navigation/navigating meet."""
    for suf in ("ations", "ation", "ings", "ing", "ments", "ment", "ness", "ions",
                "ion", "ers", "er", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)][:6]
    return w[:6]


def split_camel(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.replace("_", " ").lower().strip()
