"""
reqgraph.parser
===============

Orchestrator: text --(extractor)--> claims --(lossless tiler)--> RequirementGraph,
plus ``build_requirement`` to synthesise a brand-new requirement from elements
and ``split_requirements`` to break compound texts (two "shall" clauses) into
independent, individually parseable requirement statements.
"""

from __future__ import annotations

import re
from typing import Optional

from .core import RequirementGraph, Role
from .extractors import Extractor, RuleExtractor, _patterns
from .templates import RUPP_TEMPLATE, Template
from .tiling import tile_to_graph

# sentence/clause delimiters must be followed by whitespace so that decimals
# ("4.5 seconds") and thousands separators ("14,000 feet") never split
_SENT_BOUND = re.compile(r"[.;!?](?=\s)")
_COMMA_BOUND = re.compile(r",(?=\s)")


def _clause_boundary(text: str, lo: int, hi: int, template: Template):
    """Locate where a new requirement clause begins inside ``text[lo:hi]``,
    the window between two consecutive modality keywords.

    Returns ``(cut, resume)`` -- the previous requirement ends at ``cut`` and
    the next one starts at ``resume`` -- or None when the second modality does
    not open an independent clause (e.g. "shall ensure the operator can ...").
    """
    p = _patterns(template)
    word = p["word"]

    # 1. sentence punctuation (rightmost, so the new segment starts at the
    #    clause that actually carries the modality)
    sentence_ends = list(_SENT_BOUND.finditer(text, lo, hi))
    if sentence_ends:
        m = sentence_ends[-1]
        resume = m.end()
        while resume < hi and text[resume].isspace():
            resume += 1
        return m.end(), resume

    def subject_like(m):
        nxt = word.match(text, m.end(), hi)
        if not nxt:
            return None
        w = nxt.group(1)
        return w.lower() in template.object_determiners or w[:1].isupper()

    # 2. top-level conjunction introducing a clause with its own subject:
    #    prefer the rightmost "and"/"or" followed by a determiner/proper noun
    #    ("...the pump and the valve AND THE controller shall..."), falling
    #    back to the rightmost conjunction that has any subject text at all
    if p["conjunction"] is not None:
        best = fallback = None
        for m in p["conjunction"].finditer(text, lo, hi):
            looks = subject_like(m)
            if looks is None:
                continue
            fallback = m
            if looks:
                best = m
        m = best or fallback
        if m is not None:
            cut = m.start()
            if cut > lo and text[cut - 1] == ",":
                cut -= 1  # "...the valve, and the..." -> drop the comma too
            return cut, m.end()

    # 3. comma splice: "The system shall X, the operator shall Y"
    for m in reversed(list(_COMMA_BOUND.finditer(text, lo, hi))):
        resume = m.end()
        while resume < hi and text[resume].isspace():
            resume += 1
        if word.match(text, resume, hi):
            return m.start(), resume
    return None


def split_requirements(text: str, template: Template = RUPP_TEMPLATE) -> list[str]:
    """Split a compound text into independent requirement statements.

    A text holds multiple requirements when more than one modality keyword
    ("shall", "must", ...) appears, each in its own clause -- separated by
    sentence punctuation, a clause-level conjunction, or a comma splice.
    A single modality governing a compound action ("shall shut off the engine
    and activate the system") is atomic enough to stay one requirement.

    Each returned segment is plain original text, so it parses (and
    round-trips) like any standalone requirement.
    """
    if not text:
        return [text]
    p = _patterns(template)
    if p["modality"] is None:
        return [text]
    mods = list(p["modality"].finditer(text))
    if len(mods) < 2:
        return [text]

    cuts = []
    pos = 0  # never search before the previous boundary's resume point
    for prev, cur in zip(mods, mods[1:]):
        b = _clause_boundary(text, max(prev.end(), pos), cur.start(), template)
        if b is not None:
            cuts.append(b)
            pos = b[1]
    if not cuts:
        return [text]

    segments, start = [], 0
    for cut, resume in cuts:
        seg = text[start:cut].strip()
        if seg:
            segments.append(seg)
        start = resume
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments or [text]


class RequirementParser:
    """Parse a requirement into a graph using a chosen template + extractor."""

    def __init__(self, template: Template = RUPP_TEMPLATE,
                 extractor: Optional[Extractor] = None):
        self.template = template
        self.extractor = extractor or RuleExtractor()

    def parse(self, text: str, metadata: Optional[dict] = None) -> RequirementGraph:
        if not isinstance(text, str):
            raise TypeError(f"requirement text must be str, got {type(text).__name__}")
        claims = self.extractor.extract(text, self.template) if text else []
        return tile_to_graph(text, claims, self.template.name, metadata)

    def split(self, text: str) -> list[str]:
        """Independent requirement statements contained in ``text``."""
        if not isinstance(text, str):
            raise TypeError(f"requirement text must be str, got {type(text).__name__}")
        return split_requirements(text, self.template)

    def parse_many(self, text: str,
                   metadata: Optional[dict] = None) -> list[RequirementGraph]:
        """Split a compound text and parse each requirement separately.

        When the text splits, an ``id`` in ``metadata`` gets a ``-1``/``-2``
        suffix so every resulting graph stays uniquely identified.
        """
        segments = self.split(text)
        graphs = []
        for i, seg in enumerate(segments, 1):
            md = dict(metadata) if metadata else None
            if md and len(segments) > 1 and md.get("id"):
                md["id"] = f"{md['id']}-{i}"
            graphs.append(self.parse(seg, md))
        return graphs

    def roundtrip_ok(self, text: str) -> bool:
        return self.parse(text).generate() == text


def build_requirement(elements: dict, template: Template = RUPP_TEMPLATE,
                      metadata: Optional[dict] = None,
                      extractor: Optional[Extractor] = None) -> RequirementGraph:
    """Create a requirement graph from semantic elements (custom authoring).

    ``elements`` maps Role (or its name) -> text. The pieces are laid out in the
    template's slot order, then re-parsed so the result is a normal graph.
    """
    norm = {}
    for k, v in elements.items():
        role = k if isinstance(k, Role) else Role[str(k).upper()]
        norm[role] = v.strip()

    parts = []
    for role in template.slot_order:
        if norm.get(role):
            val = norm[role]
            if role is Role.CONDITION and not val.endswith(","):
                val += ","
            parts.append(val)
    text = " ".join(parts)
    if not text.endswith((".", "!", "?")):
        text += "."
    return RequirementParser(template, extractor).parse(text, metadata)
