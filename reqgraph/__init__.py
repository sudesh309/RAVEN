"""
reqgraph -- IREB-CPRE aligned Requirement <-> Graph toolkit.

Quick start
-----------
    from reqgraph import RequirementParser, RUPP_TEMPLATE
    g = RequirementParser(RUPP_TEMPLATE).parse(
        "When the cabin altitude exceeds 14,000 feet, the oxygen system "
        "shall deploy the passenger oxygen masks within 4 seconds.")
    print(g.summary())
    assert g.generate() == ...   # byte-exact, always

Swap the extraction backend:
    from reqgraph.extractors import SpacyExtractor, BertTaggerExtractor
    RequirementParser(RUPP_TEMPLATE, SpacyExtractor()).parse(text)

Intelligence add-ons:
    from reqgraph.nlp import BertTokenTagger, RequirementAnalyzer
    from reqgraph.quality import enrich
"""

import logging as _logging

from .errors import (ReqGraphError, GraphIntegrityError, ExtractionError,
                     TemplateError, DataFormatError, ModelError)
from .core import Edge, Node, OBLIGATION, RequirementGraph, Rel, Role
from .templates import (EARS_TEMPLATE, RUPP_TEMPLATE, Template, TEMPLATES,
                        register_template)
from .tiling import tile_to_graph
from .parser import RequirementParser, build_requirement, split_requirements
from .extractors import (Extractor, RuleExtractor, SpacyExtractor,
                         BertTaggerExtractor, auto_select, get_extractor)

# library logging etiquette: emit nothing unless the application configures it
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "Role", "Rel", "Node", "Edge", "OBLIGATION", "RequirementGraph",
    "Template", "RUPP_TEMPLATE", "EARS_TEMPLATE", "TEMPLATES", "register_template",
    "tile_to_graph", "RequirementParser", "build_requirement", "split_requirements",
    "Extractor", "RuleExtractor", "SpacyExtractor", "BertTaggerExtractor",
    "auto_select", "get_extractor",
    "ReqGraphError", "GraphIntegrityError", "ExtractionError",
    "TemplateError", "DataFormatError", "ModelError",
]

__version__ = "1.0.0"
