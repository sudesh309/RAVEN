"""
reqgraph.quality
================

IREB-flavoured requirement analysis that enriches a graph (no ML required):

* quality / ambiguity smells  -- weak words, passive voice, missing modality,
  vague quantifiers, non-atomic (compound) requirements (IREB: clarity,
  unambiguity, verifiability, atomicity).
* **completeness** -- INCOSE GtWR C4/C7: is the requirement fully stated in one
  place?  Structural roles present, no TBD placeholders, no dangling pronoun
  references, quantities carry units, performance requirements are measurable.
* requirement *type* classification (functional / performance / interface /
  safety / usability).
* EARS pattern detection (ubiquitous / event / state / unwanted / optional).

``enrich(graph)`` writes the results into ``graph.analysis``.
"""

from __future__ import annotations

import re

from .core import RequirementGraph, Role

WEAK_WORDS = [
    "fast", "slow", "quick", "quickly", "efficient", "user-friendly", "easy",
    "flexible", "robust", "approximately", "about", "roughly", "minimal",
    "maximal", "sufficient", "adequate", "appropriate", "as needed",
    "if possible", "etc", "and/or", "several", "some", "many", "few",
    "reliable", "state-of-the-art", "optimal", "seamless", "intuitive",
    "better", "improved", "normal", "as applicable", "high-quality",
]

TYPE_KEYWORDS = {
    "performance": ["within", "millisecond", "second", "minute", "rate", "throughput",
                    "latency", "accuracy", "resolution", "response time", "per second",
                    "frequency", "hz"],
    "interface": ["interface", "display", "transmit", "receive", "message", "signal",
                  "bus", "protocol", "telemetry", "packet", "report", "send", "acknowledge"],
    "safety": ["fire", "emergency", "fail", "failure", "hazard", "safe", "shutdown",
               "shut off", "warning", "alarm", "redundan", "suppression", "abort"],
    "usability": ["operator", "pilot", "crew", "user", "ability to", "configure",
                  "select", "enter"],
}


# precompiled once at import: check_quality is called per row in batch runs
_WEAK_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(w) for w in
                              sorted(WEAK_WORDS, key=len, reverse=True)) + r")(?![\w-])")
_PASSIVE_RE = re.compile(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b")
_MODALITY_RE = re.compile(r"\b(shall|should|must|will|may|can)\b")
_VAGUE_RE = re.compile(r"\b(some|several|many|few|various|a number of|as much as possible)\b")
_CONJ_RE = re.compile(r"\b(and|or)\b")
# matches multi-word negating modalities first so "shall not" counts as one
_MODALITY_COUNT_RE = re.compile(
    r"\b(shall\s+not|should\s+not|must\s+not|will\s+not|may\s+not|cannot"
    r"|shall|should|must|will|may|can)\b")


def check_quality(text: str) -> dict:
    low = text.lower()
    weak = sorted({m.group(1) for m in _WEAK_RE.finditer(low)})
    return {
        "weak_words": weak,
        "passive_voice": bool(_PASSIVE_RE.search(low)),
        "missing_modality": not _MODALITY_RE.search(low),
        "vague_quantifier": bool(_VAGUE_RE.search(low)),
        "non_atomic": len(_CONJ_RE.findall(low)) > 1,
        "compound_requirement": len(_MODALITY_COUNT_RE.findall(low)) > 1,
    }


# --- completeness (INCOSE GtWR C4 "complete", C7 "verifiable") --------------
#
# A requirement is *complete* when it is fully stated in one place: it says who
# does what, it is binding, it leaves no placeholder open, it does not lean on
# a neighbouring sentence for its meaning, and any quantity it states is
# measurable.  Each finding carries a severity so a set can be triaged:
#   blocker -- the requirement cannot be designed or verified as written
#   major   -- it is designable but not yet verifiable / self-contained
#   minor   -- it is usable; tightening it would remove residual risk
_SEVERITY_COST = {"blocker": 35, "major": 20, "minor": 8}

# TBD/TBC/TBR and friends: an explicit "we don't know yet" marker
_PLACEHOLDER_RE = re.compile(
    r"(?<![\w-])(tbd|tbc|tbr|tbs|n/a|xxx+|\?{2,}|<[^>]{0,40}>|\[[^\]]{0,40}\]"
    r"|to\s+be\s+(determined|defined|confirmed|specified|supplied|advised))"
    r"(?![\w-])", re.I)

# a requirement whose subject is a bare pronoun does not stand alone
_PRONOUN_SUBJECT_RE = re.compile(
    r"^(the\s+)?(it|this|that|they|these|those|he|she|him|her|them|"
    r"such\s+\w+|said\s+\w+|the\s+above|the\s+former|the\s+latter)$", re.I)

# engineering units — a number followed by one of these is measurable
_UNITS = (
    r"s|sec|secs|second|seconds|ms|millisecond|milliseconds|us|microsecond|"
    r"min|mins|minute|minutes|h|hr|hrs|hour|hours|day|days|"
    r"hz|khz|mhz|ghz|rpm|"
    r"m|mm|cm|km|nm|ft|feet|foot|in|inch|inches|mi|mile|miles|"
    r"kg|g|mg|t|lb|lbs|ton|tons|"
    r"n|kn|nm|psi|bar|pa|kpa|mpa|atm|"
    r"v|mv|kv|a|ma|w|kw|mw|kwh|ah|"
    r"c|f|k|deg|degree|degrees|rad|°|%|percent|"
    r"db|dba|kt|kts|knot|knots|mph|kph|km/h|m/s|g|"
    r"bit|bits|byte|bytes|kb|mb|gb|tb|bps|kbps|mbps|gbps|"
    r"px|pixel|pixels|cycle|cycles|count|item|items|time|times"
)
# a number that is followed by nothing that could describe it (end of clause or
# a connective) states a magnitude without a dimension.  The (?!\d) guards keep
# thousands separators and decimals ("14,000 feet") from looking like an end.
_BARE_NUMBER_RE = re.compile(
    r"(?<![\w.])\d+(?:[.,]\d+)?\s*"
    r"(?=$|[.,;:!?)](?!\d)|\s+(?:and|or|then|but|while|if|when)\b)", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.])\d")
# an exact value with no stated tolerance or comparator (INCOSE R21)
_TOLERANCE_RE = re.compile(
    r"(±|\+/-|\+\s*-|at\s+least|at\s+most|no\s+(more|less|later|earlier)\s+than|"
    r"greater\s+than|less\s+than|minimum|maximum|min\.|max\.|within|between|"
    r"up\s+to|not\s+to\s+exceed|nlt|ngt|>=|<=|>|<|~|tolerance|range)", re.I)
# text that stops mid-thought
_TRUNCATED_RE = re.compile(r"(,|;|:|\b(and|or|but|with|to|of|for|the|a|an))\s*$", re.I)


def _finding(key, label, severity, hint):
    return {"key": key, "label": label, "severity": severity, "hint": hint}


def check_completeness(graph_or_text, req_type: str | None = None) -> dict:
    """Structural + semantic completeness of a single requirement.

    Accepts a :class:`RequirementGraph` (enables the role checks) or raw text
    (textual checks only).  Returns a dict with ``complete``, ``score`` (0-100),
    ``missing`` (list of findings) and the individual boolean flags.
    """
    graph = None if isinstance(graph_or_text, str) else graph_or_text
    text = graph_or_text if graph is None else graph.generate()
    low = text.lower()
    findings: list[dict] = []

    # --- structural: does it say who does what, bindingly? ---
    if graph is not None:
        roles = {n.role for n in graph.elements()}
        if not (Role.SUBJECT in roles or Role.ACTOR in roles):
            findings.append(_finding(
                "no_subject", "no subject", "blocker",
                "state which system or actor is responsible"))
        if Role.MODALITY not in roles:
            findings.append(_finding(
                "no_modality", "no modality", "blocker",
                "add shall/should/may so the obligation is binding"))
        if Role.PROCESS not in roles:
            findings.append(_finding(
                "no_process", "no action", "blocker",
                "state the action the subject performs"))
        else:
            # the rules backend folds an un-splittable object into the PROCESS
            # node, so "is there an OBJECT node" says more about the backend
            # than about the text.  What is unambiguous either way: a predicate
            # of one bare word says nothing about what the action applies to.
            predicate = " ".join(
                n.text for n in graph.elements()
                if n.role in (Role.PROCESS, Role.OBJECT, Role.DETAILS,
                              Role.CONSTRAINT)).strip()
            if len(predicate.split()) < 2:
                findings.append(_finding(
                    "bare_predicate", f"action '{predicate}' has no object", "minor",
                    "say what the action operates on, or bound it with a constraint"))
        # an EARS trigger word with nothing after it leaves the case open
        for n in graph.by_role(Role.CONDITION):
            marker = (n.attrs.get("marker") or "").strip()
            body = n.text.strip()
            if marker and body.lower().replace(marker.lower(), "", 1).strip(" ,") == "":
                findings.append(_finding(
                    "open_condition", "condition states no trigger", "major",
                    f"complete the '{marker}' clause"))
                break
    else:
        if not _MODALITY_RE.search(low):
            findings.append(_finding(
                "no_modality", "no modality", "blocker",
                "add shall/should/may so the obligation is binding"))

    # --- placeholders: an explicit open item ---
    holes = sorted({m.group(1).strip() for m in _PLACEHOLDER_RE.finditer(text)})
    if holes:
        findings.append(_finding(
            "placeholder", "placeholder: " + ", ".join(holes), "blocker",
            "resolve the open value before baselining"))

    # --- self-containment: does it lean on a neighbouring sentence? ---
    subj_text = ""
    if graph is not None:
        subj = graph.by_role(Role.SUBJECT)
        subj_text = subj[0].text.strip() if subj else ""
    else:
        subj_text = text.strip().split(" shall ")[0].strip()
    if subj_text and _PRONOUN_SUBJECT_RE.match(subj_text.strip(" ,.")):
        findings.append(_finding(
            "dangling_reference", f"subject is a reference ('{subj_text.strip()}')",
            "major", "name the system explicitly so the requirement stands alone"))

    # --- measurability ---
    rtype = req_type or classify_type(text)
    has_number = bool(_NUMBER_RE.search(text))
    if rtype == "performance" and not has_number:
        findings.append(_finding(
            "unquantified_performance", "performance claim has no value", "major",
            "state the measurable threshold the design must meet"))
    bare = sorted({m.group(0).strip() for m in _BARE_NUMBER_RE.finditer(text)})
    if bare:
        findings.append(_finding(
            "number_without_unit", "value without unit: " + ", ".join(bare),
            "major", "give the quantity a dimension (s, m, %, Hz…)"))
    if has_number and not _TOLERANCE_RE.search(low) and rtype == "performance":
        findings.append(_finding(
            "no_tolerance", "exact value with no tolerance", "minor",
            "add a bound (±, at least, not to exceed…) so it can be measured"))

    # --- truncation ---
    if _TRUNCATED_RE.search(text.strip().rstrip(".")):
        findings.append(_finding(
            "truncated", "sentence stops mid-thought", "major",
            "the requirement text appears to be cut off"))

    score = max(0, 100 - sum(_SEVERITY_COST[f["severity"]] for f in findings))
    worst = ("blocker" if any(f["severity"] == "blocker" for f in findings)
             else "major" if any(f["severity"] == "major" for f in findings)
             else "minor" if findings else None)
    out = {
        "complete": not any(f["severity"] in ("blocker", "major") for f in findings),
        "score": score,
        "severity": worst,
        "missing": findings,
        "n_blockers": sum(1 for f in findings if f["severity"] == "blocker"),
    }
    for key in ("no_subject", "no_modality", "no_process", "bare_predicate",
                "open_condition", "placeholder", "dangling_reference",
                "unquantified_performance", "number_without_unit",
                "no_tolerance", "truncated"):
        out[key] = any(f["key"] == key for f in findings)
    return out


def classify_type(graph_or_text) -> str:
    text = graph_or_text if isinstance(graph_or_text, str) else graph_or_text.generate()
    low = text.lower()
    scores = {typ: sum(low.count(k) for k in kws) for typ, kws in TYPE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "functional"


def classify_ears(graph: RequirementGraph) -> str:
    markers = []
    for n in graph.by_role(Role.CONDITION):
        m = (n.attrs.get("marker") or n.text.strip().split()[0]).lower()
        markers.append(m)
    text = " ".join(markers)
    if not markers:
        return "ubiquitous"
    if any(m in ("when", "once", "after", "whenever") for m in markers) or "as soon" in text:
        return "event-driven (WHEN)"
    if any(m in ("while", "during") for m in markers) or "as long" in text:
        return "state-driven (WHILE)"
    if any(m in ("if", "unless") for m in markers) or "in case" in text:
        return "unwanted behaviour (IF/THEN)"
    if any(m in ("where", "provided", "given") for m in markers):
        return "optional feature (WHERE)"
    return "ubiquitous"


def enrich(graph: RequirementGraph) -> RequirementGraph:
    """Attach quality, completeness, type and EARS analysis to ``graph.analysis``."""
    text = graph.generate()
    graph.analysis["quality"] = check_quality(text)
    rtype = classify_type(text)
    graph.analysis["type"] = rtype
    graph.analysis["ears_pattern"] = classify_ears(graph)
    graph.analysis["completeness"] = check_completeness(graph, req_type=rtype)
    return graph
