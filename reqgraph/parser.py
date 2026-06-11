"""
reqgraph.parser
===============

Orchestrator: text --(extractor)--> claims --(lossless tiler)--> RequirementGraph,
plus ``build_requirement`` to synthesise a brand-new requirement from elements.
"""

from __future__ import annotations

from typing import Optional

from .core import RequirementGraph, Role
from .extractors import Extractor, RuleExtractor
from .templates import RUPP_TEMPLATE, Template
from .tiling import tile_to_graph


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
