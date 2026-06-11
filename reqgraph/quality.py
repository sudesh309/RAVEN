"""
reqgraph.quality
================

IREB-flavoured requirement analysis that enriches a graph (no ML required):

* quality / ambiguity smells  -- weak words, passive voice, missing modality,
  vague quantifiers, non-atomic (compound) requirements (IREB: clarity,
  unambiguity, verifiability, atomicity).
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


def check_quality(text: str) -> dict:
    low = text.lower()
    weak = sorted({m.group(1) for m in _WEAK_RE.finditer(low)})
    return {
        "weak_words": weak,
        "passive_voice": bool(_PASSIVE_RE.search(low)),
        "missing_modality": not _MODALITY_RE.search(low),
        "vague_quantifier": bool(_VAGUE_RE.search(low)),
        "non_atomic": len(_CONJ_RE.findall(low)) > 1,
    }


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
    """Attach quality, type and EARS analysis to ``graph.analysis``."""
    text = graph.generate()
    graph.analysis["quality"] = check_quality(text)
    graph.analysis["type"] = classify_type(text)
    graph.analysis["ears_pattern"] = classify_ears(graph)
    return graph
