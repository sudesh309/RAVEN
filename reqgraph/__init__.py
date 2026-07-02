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
from .corpus import (Connection, ElementRef, RequirementSetGraph,
                     build_requirement_set_graph)
from .io_formats import (read_requirements_json,)
from .sysml_parser import SysMLElement, SysMLModel, parse_sysml, read_sysml
from .sysml_compare import ComparisonReport, MatchDetail, compare as compare_sysml
from .sysml_v1_parser import (V1Element, V1Relation, SysMLV1Model,
                               parse_sysml_v1, read_sysml_v1, parse_mdzip)
from .sysml_v1_compare import (V1MatchDetail, V1ComparisonReport, OntologyDiff,
                                compare_v1, ontology_diff)
from .traceability import (VerificationMethod, TraceStatus, Allocation,
                            TraceItem, Finding, TraceabilityMatrix,
                            build_traceability_matrix, assign_verification_method)

# library logging etiquette: emit nothing unless the application configures it
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "Role", "Rel", "Node", "Edge", "OBLIGATION", "RequirementGraph",
    "Template", "RUPP_TEMPLATE", "EARS_TEMPLATE", "TEMPLATES", "register_template",
    "tile_to_graph", "RequirementParser", "build_requirement", "split_requirements",
    "Extractor", "RuleExtractor", "SpacyExtractor", "BertTaggerExtractor",
    "auto_select", "get_extractor",
    "Connection", "ElementRef", "RequirementSetGraph", "build_requirement_set_graph",
    "read_requirements_json",
    "SysMLElement", "SysMLModel", "parse_sysml", "read_sysml",
    "ComparisonReport", "MatchDetail", "compare_sysml",
    "V1Element", "V1Relation", "SysMLV1Model", "parse_sysml_v1", "read_sysml_v1",
    "parse_mdzip",
    "V1MatchDetail", "V1ComparisonReport", "OntologyDiff", "compare_v1", "ontology_diff",
    "VerificationMethod", "TraceStatus", "Allocation", "TraceItem", "Finding",
    "TraceabilityMatrix", "build_traceability_matrix", "assign_verification_method",
    "ReqGraphError", "GraphIntegrityError", "ExtractionError",
    "TemplateError", "DataFormatError", "ModelError",
]

__version__ = "1.0.0"
