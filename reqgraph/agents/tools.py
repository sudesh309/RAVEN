"""
Tool interface: the review agents as JSON-schema tools.

The same specs serve every host:

* :data:`TOOL_SPECS` — ``{"name", "description", "input_schema"}`` (the MCP /
  Anthropic / OpenAI function-calling shape);
* :func:`call_tool` — execute one by name with a JSON ``arguments`` dict;
* :func:`as_langchain_tools` — ``StructuredTool`` objects for LangChain /
  LangGraph ``ToolNode`` (needs ``langchain-core``);
* :mod:`reqgraph.agents.mcp_server` — serves them over MCP stdio.

Inputs accept the requirement set inline (``requirements``: list of
``{"id", "text", ...metadata}``) or as a file (``requirements_path``: CSV,
Excel, JSON, ReqIF, Word, TXT), and the architecture inline
(``architecture_text``: SysML v2) or as a file (``architecture_path``: .sysml,
.xmi, .ttl, .mdzip).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..errors import DataFormatError
from .orchestrator import AGENT_CLASSES, TEAMS, default_agents, ReviewOrchestrator

_REQ_ITEM = {"type": "object",
             "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                            "parent": {"type": "string", "description": "parent requirement id(s)"},
                            "level": {"type": "string"}, "rationale": {"type": "string"},
                            "verification_method": {"type": "string"},
                            "allocated_to": {"type": "string",
                                             "description": "architecture element(s)"}},
             "required": ["text"], "additionalProperties": {"type": "string"}}
_REQ_INPUT = {
    "requirements": {"type": "array", "items": _REQ_ITEM,
                     "description": "the requirement set, with any metadata as extra keys"},
    "requirements_path": {"type": "string",
                          "description": "or a file: .csv .xlsx .json .reqif .docx .txt"},
}
_ARCH_INPUT = {
    "architecture_text": {"type": "string", "description": "SysML v2 textual model"},
    "architecture_path": {"type": "string",
                          "description": "or a model file: .sysml (v2), .xmi/.ttl/.mdzip (v1)"},
}
_LLM_INPUT = {
    "use_llm": {"type": "boolean", "default": False,
                "description": "let LLM-capable agents add candidate findings"},
    "llm_provider": {"type": "string", "enum": ["ollama", "vertex"]},
    "llm_model": {"type": "string"},
}
_FORMAT = {"format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}}

TOOL_SPECS = [
    {"name": "raven_review",
     "description": ("Full multi-agent review of a requirement set: writing quality, "
                     "completeness, rationale/verification, consistency, requirement "
                     "cascade, and — when an architecture model is given — allocation, "
                     "architecture justification and derived requirements. Returns a "
                     "per-requirement scorecard and severity-ranked findings."),
     "input_schema": {"type": "object", "properties": {
         **_REQ_INPUT, **_ARCH_INPUT, **_LLM_INPUT, **_FORMAT,
         "agents": {"type": "array", "items": {"type": "string"},
                    "description": "agent or team names (quality, cascade, architecture); default all"}}}},
    {"name": "raven_quality_review",
     "description": ("Requirement quality team only (no architecture): INCOSE wording "
                     "rules, completeness, rationale / source / verification method, "
                     "duplicates and conflicts."),
     "input_schema": {"type": "object", "properties": {**_REQ_INPUT, **_LLM_INPUT, **_FORMAT}}},
    {"name": "raven_cascade_review",
     "description": ("Requirement cascade check from parent links and levels: orphans, "
                     "dangling links, cycles, level inversions, undecomposed requirements, "
                     "children that do not cover their parent, child budgets looser than "
                     "the parent's."),
     "input_schema": {"type": "object", "properties": {**_REQ_INPUT, **_LLM_INPUT, **_FORMAT}}},
    {"name": "raven_architecture_review",
     "description": ("Requirement ↔ architecture check (needs a SysML model): "
                     "requirements with no realising element, elements no requirement "
                     "justifies, model requirements missing from the spec, missing derived "
                     "requirements, child requirements allocated outside their parent's branch."),
     "input_schema": {"type": "object", "properties": {**_REQ_INPUT, **_ARCH_INPUT,
                                                       **_LLM_INPUT, **_FORMAT}}},
    {"name": "raven_parse_architecture",
     "description": ("Extract the structure of a SysML model: definitions, composition "
                     "tree, embedded requirements and satisfy / allocate / verify links."),
     "input_schema": {"type": "object", "properties": {**_ARCH_INPUT}}},
    {"name": "raven_list_agents",
     "description": "List the review agents, their team and what each checks.",
     "input_schema": {"type": "object", "properties": {}}},
]
_TOOL_AGENTS = {"raven_quality_review": ["quality"], "raven_cascade_review": ["cascade"],
                "raven_architecture_review": ["architecture"]}


# ---------------------------------------------------------------------------
def load_requirements(path: str) -> list:
    """Read a requirement file into ``(id, text, meta)`` triples (raises, never exits)."""
    from ..io_formats import (read_reqif, read_requirements_csv, read_requirements_docx,
                              read_requirements_excel, read_requirements_json)
    if not os.path.isfile(path):
        raise DataFormatError(f"requirements file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, encoding="utf-8") as fh:
            return [(f"R{i}", ln.strip(), {}) for i, ln in enumerate(fh, 1) if ln.strip()]
    readers = {".csv": read_requirements_csv, ".json": read_requirements_json,
               ".xlsx": read_requirements_excel, ".xls": read_requirements_excel,
               ".reqif": read_reqif, ".xml": read_reqif, ".docx": read_requirements_docx}
    if ext not in readers:
        raise DataFormatError(f"unsupported requirements file type {ext!r}")
    return readers[ext](path)


def _requirements(args) -> list:
    if args.get("requirements"):
        return args["requirements"]
    if args.get("requirements_path"):
        return load_requirements(args["requirements_path"])
    raise DataFormatError("give 'requirements' (list) or 'requirements_path'")


def _architecture(args):
    from ..architecture import load_architecture
    if args.get("architecture_text"):
        return load_architecture(args["architecture_text"], fmt=args.get("architecture_format"))
    if args.get("architecture_path"):
        if not os.path.isfile(args["architecture_path"]):
            raise DataFormatError(f"architecture file not found: {args['architecture_path']}")
        return load_architecture(args["architecture_path"], fmt=args.get("architecture_format"))
    return None


def _llm(args, llm):
    want = args.get("use_llm")
    if want is False:
        return None
    if llm is not None:
        return llm
    if want:
        from ..llm import get_provider
        return get_provider(args.get("llm_provider"), model=args.get("llm_model"))
    return None


def call_tool(name: str, arguments: Optional[dict] = None, *, llm=None) -> Any:
    """Run a tool; returns a JSON-serialisable dict (or Markdown string)."""
    args = dict(arguments or {})
    if name == "raven_list_agents":
        return {"agents": [c().describe() for c in AGENT_CLASSES], "teams": TEAMS}
    if name == "raven_parse_architecture":
        m = _architecture(args)
        if m is None:
            raise DataFormatError("give 'architecture_text' or 'architecture_path'")
        return {"summary": m.summary(), "depth": m.depth(), "model": m.to_dict()}
    if name not in ("raven_review",) + tuple(_TOOL_AGENTS):
        raise KeyError(f"unknown tool {name!r}")
    provider = _llm(args, llm)
    names = _TOOL_AGENTS.get(name) or args.get("agents")
    orch = ReviewOrchestrator(default_agents(provider, names), llm=provider,
                              config={"use_llm": bool(provider)})
    report = orch.run(_requirements(args), _architecture(args))
    if args.get("format") == "markdown":
        return report.to_markdown()
    return report.to_dict()


def as_langchain_tools(llm=None) -> list:
    """The tools as LangChain ``StructuredTool`` objects (for agents / ToolNode)."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:                        # pragma: no cover - optional
        raise ImportError("langchain-core is not installed: pip install langchain-core") from exc
    tools = []
    for spec in TOOL_SPECS:
        def run(_name=spec["name"], **kwargs):
            out = call_tool(_name, kwargs, llm=llm)
            return out if isinstance(out, str) else json.dumps(out, default=str)
        tools.append(StructuredTool.from_function(
            func=run, name=spec["name"], description=spec["description"],
            args_schema=spec["input_schema"]))
    return tools
