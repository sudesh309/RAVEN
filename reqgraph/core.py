"""
reqgraph.core
=============

Graph data model for a requirement: typed *element* nodes and typed
*relationship* edges, plus exporters (JSON / Mermaid / DOT / Cypher / networkx).

This module is backend-agnostic: it knows nothing about how the elements were
extracted (rules / spaCy / BERT). The lossless guarantee lives here, in the
``leaf_order`` backbone -- concatenating the terminal nodes' text in order
reproduces the source requirement byte-for-byte.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import DataFormatError, GraphIntegrityError

logger = logging.getLogger(__name__)


def _mermaid_label(s: str) -> str:
    """Mermaid labels cannot contain double quotes or newlines."""
    return s.replace('"', "'").replace("\n", " ").strip()


def _dot_label(s: str) -> str:
    """Graphviz DOT string escaping."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _xml_escape(s: str) -> str:
    """Escape text for an XML element/attribute body (GraphML)."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _ttl_literal(s: str) -> str:
    """Escape a string for a Turtle double-quoted literal."""
    return (s.replace("\\", "\\\\").replace('"', '\\"')
             .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


class Role(str, Enum):
    """Semantic role of a requirement element, aligned with IREB/Rupp."""
    ROOT = "ROOT"
    CONDITION = "CONDITION"     # logical/temporal pre-/post-condition
    SUBJECT = "SUBJECT"         # system-under-consideration / responsible actor
    MODALITY = "MODALITY"       # legal obligation: shall/should/will/may/must
    ACTION = "ACTION"           # one functional activity (group node)
    ACTOR = "ACTOR"             # the "whom" of a user-interaction requirement
    PROCESS = "PROCESS"         # the process verb
    OBJECT = "OBJECT"           # object the process operates on
    DETAILS = "DETAILS"         # additional object details
    CONSTRAINT = "CONSTRAINT"   # quality/performance constraint
    OPERATOR = "OPERATOR"       # logical connective AND / OR / NOT (group node)
    GLUE = "GLUE"               # literal connective/separator text (round-trip)


class Rel(str, Enum):
    HAS_CONDITION = "HAS_CONDITION"
    HAS_SUBJECT = "HAS_SUBJECT"
    HAS_MODALITY = "HAS_MODALITY"
    HAS_ACTION = "HAS_ACTION"
    HAS_ACTOR = "HAS_ACTOR"
    HAS_PROCESS = "HAS_PROCESS"
    ACTS_ON = "ACTS_ON"
    HAS_DETAILS = "HAS_DETAILS"
    HAS_CONSTRAINT = "HAS_CONSTRAINT"
    OPERAND = "OPERAND"
    NEXT = "NEXT"               # ordered terminal sequence (text order)


# IREB: the modal verb fixes the legal obligation (binding force).
OBLIGATION = {
    "shall": "mandatory (legally binding)",
    "must": "mandatory (legally binding)",
    "shall not": "prohibition (legally binding)",
    "must not": "prohibition (legally binding)",
    "should": "recommended / desirable",
    "should not": "discouraged",
    "will": "intention / future fact",
    "will not": "intention (negated)",
    "may": "optional / permitted",
    "can": "optional / capability",
    "may not": "not permitted",
    "cannot": "not permitted",
}

# terminal (text-bearing) roles
TERMINAL_ROLES = frozenset({
    Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR, Role.PROCESS,
    Role.OBJECT, Role.DETAILS, Role.CONSTRAINT, Role.GLUE,
})


@dataclass
class Node:
    id: str
    role: Role
    text: str = ""
    attrs: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.role in TERMINAL_ROLES


@dataclass
class Edge:
    source: str
    target: str
    rel: Rel
    attrs: dict = field(default_factory=dict)


class RequirementGraph:
    """Typed, ordered graph for one requirement.

    ``leaf_order`` lists the ids of all terminal nodes in exact text order;
    concatenating their text reproduces the original requirement (lossless).
    """

    def __init__(self, template_name: str = "", metadata: Optional[dict] = None):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.leaf_order: list[str] = []
        self.root_id: Optional[str] = None
        self.template_name = template_name
        self.metadata = metadata or {}
        self.analysis: dict = {}          # quality / type / EARS enrichment
        self._counter = 0

    # -- construction ---------------------------------------------------------
    def new_id(self, role: Role) -> str:
        self._counter += 1
        return f"{role.value.lower()}_{self._counter}"

    def add_node(self, role: Role, text: str = "", **attrs) -> Node:
        node = Node(self.new_id(role), role, text, dict(attrs))
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target, rel: Rel, **attrs) -> Edge:
        s = source.id if isinstance(source, Node) else source
        t = target.id if isinstance(target, Node) else target
        if s not in self.nodes or t not in self.nodes:
            raise GraphIntegrityError(
                f"edge {s!r} -[{rel.value}]-> {t!r} references an unknown node")
        edge = Edge(s, t, rel, dict(attrs))
        self.edges.append(edge)
        return edge

    # -- generation (graph -> text), guaranteed lossless ----------------------
    def generate(self) -> str:
        return "".join(self.nodes[nid].text for nid in self.leaf_order)

    # -- integrity ------------------------------------------------------------
    def verify(self) -> bool:
        """Check structural invariants; raise GraphIntegrityError on violation.

        Invariants: every leaf_order id exists and is a terminal; every
        terminal node appears exactly once in leaf_order (the tiling); all
        edges reference existing nodes; root_id (if set) exists.
        """
        seen = set()
        for nid in self.leaf_order:
            node = self.nodes.get(nid)
            if node is None:
                raise GraphIntegrityError(f"leaf_order references unknown node {nid!r}")
            if not node.is_terminal:
                raise GraphIntegrityError(f"non-terminal {nid!r} in leaf_order")
            if nid in seen:
                raise GraphIntegrityError(f"node {nid!r} tiled twice")
            seen.add(nid)
        for n in self.nodes.values():
            if n.is_terminal and n.id not in seen:
                raise GraphIntegrityError(f"terminal {n.id!r} missing from leaf_order")
        for e in self.edges:
            if e.source not in self.nodes or e.target not in self.nodes:
                raise GraphIntegrityError(
                    f"edge {e.source!r}->{e.target!r} references an unknown node")
        if self.root_id is not None and self.root_id not in self.nodes:
            raise GraphIntegrityError(f"root_id {self.root_id!r} is not a node")
        return True

    # -- queries --------------------------------------------------------------
    def elements(self) -> list[Node]:
        """Semantic (non-glue) terminal elements in text order."""
        return [self.nodes[i] for i in self.leaf_order
                if self.nodes[i].role is not Role.GLUE and self.nodes[i].text.strip()]

    def by_role(self, role: Role) -> list[Node]:
        return [n for n in self.elements() if n.role is role]

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "template": self.template_name,
            "metadata": self.metadata,
            "analysis": self.analysis,
            "root": self.root_id,
            "leaf_order": self.leaf_order,
            "nodes": [{"id": n.id, "role": n.role.value, "text": n.text, "attrs": n.attrs}
                      for n in self.nodes.values()],
            "edges": [{"source": e.source, "target": e.target, "rel": e.rel.value,
                       "attrs": e.attrs} for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RequirementGraph":
        try:
            g = cls(d.get("template", ""), d.get("metadata", {}))
            g.analysis = d.get("analysis", {})
            for nd in d["nodes"]:
                g.nodes[nd["id"]] = Node(nd["id"], Role(nd["role"]),
                                         nd.get("text", ""), nd.get("attrs", {}))
            for ed in d["edges"]:
                g.edges.append(Edge(ed["source"], ed["target"], Rel(ed["rel"]),
                                    ed.get("attrs", {})))
            g.leaf_order = d["leaf_order"]
            g.root_id = d.get("root")
        except (KeyError, ValueError, TypeError) as exc:
            raise DataFormatError(f"malformed graph payload: {exc}") from exc
        g.verify()
        return g

    # -- exporters ------------------------------------------------------------
    def to_mermaid(self, show_glue: bool = False) -> str:
        lines = ["flowchart TD"]
        shapes = {Role.ROOT: ("([", "])"), Role.OPERATOR: ("{{", "}}"),
                  Role.ACTION: ("[/", "/]")}
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            label = n.role.value if not n.text.strip() else f"{n.role.value}: {n.text.strip()}"
            o, c = shapes.get(n.role, ("[", "]"))
            lines.append(f'    {n.id}{o}"{_mermaid_label(label)}"{c}')
        for e in self.edges:
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and self.nodes[e.target].role is Role.GLUE:
                continue
            lines.append(f"    {e.source} -->|{e.rel.value}| {e.target}")
        return "\n".join(lines)

    def to_dot(self, show_glue: bool = False) -> str:
        lines = ["digraph Requirement {", "  rankdir=TB;", '  node [fontname="Helvetica"];']
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            label = n.role.value if not n.text.strip() else f"{n.role.value}\n{n.text.strip()}"
            lines.append(f'  "{n.id}" [label="{_dot_label(label)}"];')
        for e in self.edges:
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and self.nodes[e.target].role is Role.GLUE:
                continue
            lines.append(f'  "{e.source}" -> "{e.target}" [label="{e.rel.value}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_cypher(self) -> str:
        stmts = []
        for n in self.nodes.values():
            if n.role is Role.GLUE:
                continue
            props = {"id": n.id, "text": n.text.strip(), **n.attrs}
            prop_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
            stmts.append(f"CREATE (:{n.role.value} {{{prop_str}}})")
        for e in self.edges:
            if e.rel is Rel.NEXT or self.nodes[e.target].role is Role.GLUE:
                continue
            stmts.append(f"MATCH (a {{id:'{e.source}'}}),(b {{id:'{e.target}'}}) "
                         f"CREATE (a)-[:{e.rel.value}]->(b)")
        return "\n".join(stmts)

    def to_networkx(self):
        import networkx as nx
        g = nx.DiGraph(template=self.template_name, **self.metadata)
        for n in self.nodes.values():
            g.add_node(n.id, role=n.role.value, text=n.text, **n.attrs)
        for e in self.edges:
            g.add_edge(e.source, e.target, rel=e.rel.value, **e.attrs)
        return g

    def to_graphml(self, show_glue: bool = False) -> str:
        """GraphML (XML) knowledge graph — readable by yEd, Gephi, networkx, Cytoscape.

        Node attributes (``role``, ``text`` and any extras as a JSON ``attrs``
        string) and the edge ``rel`` are declared as typed ``<key>`` elements so
        the file is valid GraphML. Built with the standard library only.
        """
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 f'<graphml xmlns="{ns}">',
                 '  <key id="role" for="node" attr.name="role" attr.type="string"/>',
                 '  <key id="text" for="node" attr.name="text" attr.type="string"/>',
                 '  <key id="attrs" for="node" attr.name="attrs" attr.type="string"/>',
                 '  <key id="rel" for="edge" attr.name="rel" attr.type="string"/>',
                 f'  <graph id="{_xml_escape(self.template_name or "requirement")}" '
                 'edgedefault="directed">']
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            lines.append(f'    <node id="{_xml_escape(n.id)}">')
            lines.append(f'      <data key="role">{_xml_escape(n.role.value)}</data>')
            if n.text.strip():
                lines.append(f'      <data key="text">{_xml_escape(n.text.strip())}</data>')
            if n.attrs:
                lines.append('      <data key="attrs">'
                             f'{_xml_escape(json.dumps(n.attrs, ensure_ascii=False))}</data>')
            lines.append('    </node>')
        for i, e in enumerate(self.edges):
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and self.nodes[e.target].role is Role.GLUE:
                continue
            lines.append(f'    <edge id="e{i}" source="{_xml_escape(e.source)}" '
                         f'target="{_xml_escape(e.target)}">')
            lines.append(f'      <data key="rel">{_xml_escape(e.rel.value)}</data>')
            lines.append('    </edge>')
        lines.append('  </graph>')
        lines.append('</graphml>')
        return "\n".join(lines)

    def to_turtle(self, base: str = "http://reqgraph.org/ns#",
                  show_glue: bool = False) -> str:
        """RDF Turtle (.ttl) knowledge graph — loadable into any triple store.

        Each element becomes a typed resource (``a rg:<Role>``) carrying its text
        as ``rdfs:label`` / ``rg:text`` plus any attributes as ``rg:<attr>``;
        relationships become predicates (``rg:<Rel>``) between resources.
        """
        prefix = "rg"
        lines = [f"@prefix {prefix}: <{base}> .",
                 "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                 ""]
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            triples = [f"a {prefix}:{n.role.value}"]
            if n.text.strip():
                lit = f'"{_ttl_literal(n.text.strip())}"'
                triples.append(f"rdfs:label {lit}")
                triples.append(f"{prefix}:text {lit}")
            for k, v in n.attrs.items():
                triples.append(f'{prefix}:{k} "{_ttl_literal(str(v))}"')
            body = " ;\n    ".join(triples)
            lines.append(f"{prefix}:{n.id} {body} .")
        lines.append("")
        for e in self.edges:
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and self.nodes[e.target].role is Role.GLUE:
                continue
            lines.append(f"{prefix}:{e.source} {prefix}:{e.rel.value} {prefix}:{e.target} .")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = []
        if self.metadata:
            lines.append("Attributes: " + ", ".join(f"{k}={v}" for k, v in self.metadata.items()))
        lines.append(f"Template  : {self.template_name}")
        if self.analysis:
            for k in ("type", "ears_pattern"):
                if k in self.analysis:
                    lines.append(f"{k:<10}: {self.analysis[k]}")
            sm = self.analysis.get("quality", {})
            if sm:
                flags = [k for k, v in sm.items() if v and k != "weak_words"]
                if sm.get("weak_words"):
                    flags.append(f"weak_words={sm['weak_words']}")
                if flags:
                    lines.append("quality   : " + "; ".join(flags))
        lines.append("Elements  :")
        for n in self.elements():
            extra = ""
            if n.role is Role.MODALITY:
                extra = f"   [{n.attrs.get('obligation', '')}]"
            elif n.attrs.get("function_type"):
                extra = f"   [function: {n.attrs['function_type']}]"
            elif n.attrs.get("kind"):
                extra = f"   [{n.attrs['kind']}]"
            lines.append(f"   - {n.role.value:<11}: {n.text.strip()!r}{extra}")
        return "\n".join(lines)
