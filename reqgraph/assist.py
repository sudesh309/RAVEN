"""
reqgraph.assist
===============

LLM-assisted features, built so that the model is never the final authority:

``suggest_rewrite``
    Asks the model for an INCOSE/EARS-conformant rewrite of one requirement,
    then **re-scores the rewrite with RAVEN's own deterministic quality and
    completeness checks** and reports what was resolved or introduced. The model
    is told never to invent values; where one is needed it writes ``<value>``,
    which the verdict reports as ``needs_input`` rather than as a regression.

``refine_pluscal``
    Asks the model to improve a generated PlusCal model (e.g. a richer
    environment), then **guards against weakening**: every property the
    requirement produced must still be checked, rewritten property definitions
    are flagged for review, and the candidate is re-validated with TLC when the
    TLA+ tools are installed. A model that makes TLC pass by deleting
    ``Response`` is rejected, not accepted.

Both return a dict with ``status: "candidate"`` -- a human decides.
"""

from __future__ import annotations

import re
from typing import Optional

from .llm import LLMError, LLMProvider, extract_json
from .parser import RequirementParser
from .quality import enrich, quality_score
from .templates import RUPP_TEMPLATE, Template


# ---------------------------------------------------------------------------
# rewrite suggestions
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """You are a senior requirements engineer. Rewrite ONE system \
requirement so it conforms to the INCOSE Guide to Writing Requirements and, when \
a trigger or state applies, an EARS pattern (While <state>, / When <trigger>, / \
If <condition>, then / Where <feature>,).
Rules:
- Keep the original meaning, subject and level of obligation.
- Active voice, an explicit named subject, and "shall" (or "shall not").
- One requirement per sentence: split a compound requirement into several.
- No vague terms (fast, user-friendly, adequate, approximately, as appropriate...).
- NEVER invent numbers, units, tolerances or names. Where a value is needed but \
not given, write the placeholder <value> and list it under assumptions.
Reply with JSON only:
{"rewrite": ["<requirement sentence>", ...], "rationale": "<one short paragraph>", \
"assumptions": ["<assumption>", ...]}"""


_PLACEHOLDER_RE = re.compile(r"<\s*value\s*>", re.I)


def assess(text: str, *, template: Template = RUPP_TEMPLATE, extractor=None) -> dict:
    """Deterministic RAVEN verdict for one requirement (the re-scoring yardstick)."""
    g = enrich(RequirementParser(template, extractor).parse(text))
    q_score, smells = quality_score(g.analysis["quality"])
    comp = g.analysis["completeness"]
    return {
        "text": text,
        "quality_score": q_score,
        "smells": smells,
        "completeness_score": comp["score"],
        "complete": comp["complete"],
        "missing": [f["label"] for f in comp["missing"]],
        "finding_keys": sorted({f"q:{s.split(':')[0]}" for s in smells}
                               | {f"c:{f['key']}" for f in comp["missing"]}),
        "type": g.analysis["type"],
        "ears_pattern": g.analysis["ears_pattern"],
    }


def _rewrite_prompt(before: dict) -> str:
    lines = [f"Requirement:\n{before['text']}", ""]
    if before["smells"] or before["missing"]:
        lines.append("Findings from the automated checker:")
        lines += [f"- quality: {s}" for s in before["smells"]]
        lines += [f"- completeness: {m}" for m in before["missing"]]
    else:
        lines.append("The automated checker found no defects; improve clarity only "
                     "if it genuinely helps, otherwise return it unchanged.")
    lines.append(f"(classified as {before['type']}, EARS: {before['ears_pattern']})")
    return "\n".join(lines)


def suggest_rewrite(text: str, provider: LLMProvider, *,
                    template: Template = RUPP_TEMPLATE, extractor=None) -> dict:
    """LLM rewrite of one requirement, re-scored by RAVEN (a candidate only)."""
    if provider is None:
        raise LLMError("no LLM provider configured")
    text = (text or "").strip()
    if not text:
        raise LLMError("nothing to rewrite")
    before = assess(text, template=template, extractor=extractor)
    reply = provider.complete(_rewrite_prompt(before), system=REWRITE_SYSTEM,
                              json_mode=True, temperature=0.2)
    data = extract_json(reply)
    raw = data.get("rewrite")
    sentences = [raw] if isinstance(raw, str) else list(raw or [])
    sentences = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
    if not sentences:
        raise LLMError("the model returned no rewrite")

    rewrites = [assess(s, template=template, extractor=extractor) for s in sentences]
    after_keys = set().union(*(set(r["finding_keys"]) for r in rewrites))
    before_keys = set(before["finding_keys"])
    resolved = sorted(before_keys - after_keys)
    introduced = sorted(after_keys - before_keys)
    # A <value> placeholder also trips findings that merely follow from the
    # missing number (e.g. "performance claim has no value"). Re-check with a
    # stand-in value: if that version introduces nothing, the only gap left is
    # the engineer's value -- not a regression.
    filled_new = set()
    if "c:placeholder" in introduced:
        for s in sentences:
            filled = assess(_PLACEHOLDER_RE.sub("1", s), template=template,
                            extractor=extractor)
            filled_new |= set(filled["finding_keys"]) - before_keys
    if "c:placeholder" in introduced and not filled_new:
        verdict = "needs_input"      # the rewrite asks the engineer for a value
    elif introduced:
        verdict = "regressed"
    elif resolved:
        verdict = "improved"
    else:
        verdict = "unchanged"

    return {
        "status": "candidate",
        "verdict": verdict,
        "original": before,
        "rewrites": rewrites,
        "rationale": str(data.get("rationale", "")).strip(),
        "assumptions": [str(a) for a in (data.get("assumptions") or [])],
        "resolved": [k.split(":", 1)[1] for k in resolved],
        "introduced": [k.split(":", 1)[1] for k in introduced],
        "provider": provider.describe(),
        "review_note": "Scores are RAVEN's own checks; they cannot confirm that the "
                       "meaning was preserved -- review before adopting.",
    }


# ---------------------------------------------------------------------------
# PlusCal refinement
# ---------------------------------------------------------------------------

PLUSCAL_SYSTEM = """You are a formal-methods engineer fluent in TLA+ and PlusCal \
(P-syntax). You refine a machine-generated PlusCal reference model of ONE \
requirement. Rules:
- Keep the MODULE name and the algorithm name exactly.
- Keep EVERY property already listed in the TLC config, with the same name and \
the same meaning. You may add properties; never remove or weaken one.
- Improve only what the requirement text justifies (e.g. let a state condition \
change over time, model the environment more faithfully). Do not add behaviour \
the requirement does not mention.
- Use P-syntax PlusCal and only standard modules (Naturals, Integers, \
Sequences, FiniteSets, TLC). Keep the state space finite and small.
Reply with exactly: a ```tla fenced block with the whole module, a ```cfg \
fenced block with the whole TLC config, then one short paragraph of rationale."""

_FENCE_RE = re.compile(r"```[ \t]*([\w+\-]*)[ \t]*\n(.*?)```", re.S)
_BASELINE_DEF_RE = re.compile(r"^\s{4}([A-Z]\w*)\s*==", re.M)
_CFG_NAMES_RE = re.compile(r"^\s*(INVARIANTS?|PROPERTY|PROPERTIES)\s+(.+)$", re.M)


def _cfg_checked_names(cfg: str) -> set:
    names = set()
    for _, rest in _CFG_NAMES_RE.findall(cfg):
        names.update(re.findall(r"[A-Za-z_]\w*", rest.split("\\*")[0]))
    return names


def _definition(tla: str, name: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(name)}\s*==\s*(.+)$", tla, re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def _split_reply(reply: str) -> tuple[str, str, str]:
    tla = cfg = ""
    for lang, body in _FENCE_RE.findall(reply):
        lang = lang.lower()
        if not tla and ("MODULE" in body and "--algorithm" in body or lang in ("tla", "tla+", "pluscal")):
            tla = body.strip() + "\n"
        elif not cfg and (lang == "cfg" or "SPECIFICATION" in body):
            cfg = body.strip() + "\n"
    rationale = _FENCE_RE.sub("", reply).strip()
    return tla, cfg, rationale


def refine_pluscal(spec, provider: LLMProvider, *, validate: bool = True,
                   jar: Optional[str] = None) -> dict:
    """LLM refinement of a :class:`~reqgraph.pluscal.PlusCalSpec`, guarded.

    ``accepted`` is only a recommendation: True when no baseline property was
    dropped, no static issue was found and (if run) TLC reports no error.
    Modified property definitions are listed for a human to confirm.
    """
    from .pluscal import validate_spec

    if provider is None:
        raise LLMError("no LLM provider configured")
    checked = [p for p in spec.properties if p.get("checked")]
    prompt = "\n".join([
        f"Requirement ({spec.module}):", spec.requirement, "",
        "Element -> model mapping:",
        *[f"- {m['role']}: \"{m['text']}\" -> {m['spec']} ({m['detail']})"
          for m in spec.mapping],
        "",
        "Properties that must be kept (name: meaning):",
        *[f"- {p['name']}: {p['meaning']}" for p in checked],
        "",
        *(["Open notes:", *[f"- {n}" for n in spec.notes], ""] if spec.notes else []),
        "Current module:", "```tla", spec.tla.rstrip(), "```",
        "Current TLC config:", "```cfg", spec.cfg.rstrip(), "```",
    ])
    reply = provider.complete(prompt, system=PLUSCAL_SYSTEM, temperature=0.1)
    tla, cfg, rationale = _split_reply(reply)

    issues = []
    if not tla:
        issues.append("the reply contained no TLA+ module")
    if not cfg:
        issues.append("the reply contained no TLC config")
    if tla and not re.search(rf"-{{4,}}\s*MODULE\s+{re.escape(spec.module)}\b", tla):
        issues.append(f"module must still be named {spec.module}")
    if tla and "--algorithm" not in tla:
        issues.append("no PlusCal '--algorithm' block")

    baseline = {p["name"] for p in checked}
    kept = _cfg_checked_names(cfg) if cfg else set()
    # with no config there is nothing to compare; the issue above says so
    dropped = sorted(baseline - kept) if cfg else []
    # Compare every baseline definition, not just the property lines: a model
    # can weaken Response without touching it by redefining Done == TRUE.
    # TypeOK is expected to grow when variables are added, so it is reported
    # but does not block acceptance.
    modified, blocking = [], []
    for name in (_BASELINE_DEF_RE.findall(spec.tla) if tla else []):
        old = _definition(spec.tla, name)
        new = _definition(tla, name) if tla else None
        if new is None:
            if name in baseline or name in ("Trigger", "Done", "Pending"):
                issues.append(f"definition {name} was removed")
            continue
        if new != old:
            modified.append({"name": name, "before": old, "after": new})
            if name != "TypeOK":
                blocking.append(name)

    validation = None
    if validate and tla and cfg and not issues:
        validation = validate_spec(tla, jar, cfg=cfg)

    tlc_ok = (validation is None or not validation["available"]
              or validation.get("ok"))
    accepted = not dropped and not issues and not blocking and bool(tlc_ok)
    reasons = []
    if dropped:
        reasons.append(f"dropped required propert{'y' if len(dropped) == 1 else 'ies'}: "
                       f"{', '.join(dropped)} -- a model can pass TLC by checking less")
    if blocking:
        reasons.append(f"changed the definition of {', '.join(blocking)} -- review "
                       f"whether the requirement's meaning is preserved")
    reasons += issues
    if validation and validation["available"] and not validation.get("ok"):
        reasons.append("TLC: " + (", ".join(f"{v['kind']} {v['name']}"
                                             for v in validation["violations"])
                                  or validation.get("error") or "did not pass"))
    return {
        "status": "candidate",
        "accepted": accepted,
        "reasons": reasons,
        "tla": tla, "cfg": cfg, "rationale": rationale,
        "dropped_properties": dropped,
        "modified_properties": modified,
        "validation": validation,
        "provider": provider.describe(),
    }
