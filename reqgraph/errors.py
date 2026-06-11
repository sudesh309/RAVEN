"""
reqgraph.errors
===============

Exception hierarchy. Everything raised intentionally by reqgraph derives from
``ReqGraphError`` so callers can catch one type at an application boundary
(the CLI does exactly that).
"""


class ReqGraphError(Exception):
    """Base class for all reqgraph errors."""


class GraphIntegrityError(ReqGraphError):
    """The graph violates a structural invariant (tiling, dangling edge...)."""


class ExtractionError(ReqGraphError):
    """An extraction backend failed or is unavailable."""


class TemplateError(ReqGraphError):
    """A template is malformed or unknown."""


class DataFormatError(ReqGraphError):
    """An input/output file or payload has the wrong shape."""


class ModelError(ReqGraphError):
    """A machine-learning model could not be loaded, trained, or applied."""
