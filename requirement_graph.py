"""
requirement_graph.py
=====================

Convert any textual *requirement* into a semantic **graph** and back again,
following IREB-CPRE requirements-engineering concepts.

Design goals (as a requirements engineer would phrase them)
-----------------------------------------------------------
R1  A requirement SHALL be decomposed into typed *elements* (nodes) and typed
    *relationships* (edges) that follow the IREB / Rupp "requirements template"
    (a.k.a. MASTeR boilerplate).
R2  The graph SHALL be *semantically equivalent* to the source requirement.
R3  The system SHALL be able to regenerate the *exact* original requirement
    text from the graph (lossless round-trip).
R4  The user SHALL be able to define *custom* requirement structures (templates)
    without changing the engine.

How the three goals are guaranteed at the same time
---------------------------------------------------
The original sentence is *tiled*: every character is owned by exactly one
terminal node. Semantic elements (CONDITION, SUBJECT, MODALITY, PROCESS,
OBJECT, ...) own their meaningful spans; the connective/separator characters in
between are owned by GLUE nodes. Regeneration is therefore just an ordered
concatenation of the terminal nodes -> the output is *byte-for-byte identical*
to the input (R3), no matter how good or bad the semantic classification was.

On top of those terminals sits a semantic tree (ROOT -> ACTION -> PROCESS/OBJECT,
logical AND/OR operators for compound actions, conditions, constraints) that
carries the meaning (R2) and is what you visualise / query / edit.

Templates (R1, R4) are pure configuration: keyword sets + slot order. Two are
shipped: the IREB/Rupp boilerplate and EARS (Easy Approach to Requirements
Syntax, widely used in aerospace). You register your own with one dataclass.

The module has **no third-party dependencies**. Optional exporters to
networkx / Neo4j-Cypher / Mermaid / Graphviz-DOT / JSON are provided.

Author: requirements-engineering reference implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Vocabulary: element (node) roles and relationship (edge) types
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Semantic role of a requirement element, aligned with IREB/Rupp."""
    ROOT = "ROOT"                 # the requirement as a whole
    CONDITION = "CONDITION"       # logical/temporal pre-condition (When/If/While...)
    SUBJECT = "SUBJECT"           # the system-under-consideration / responsible actor
    MODALITY = "MODALITY"         # legal obligation: shall/should/will/may/must
    ACTION = "ACTION"             # a single functional activity (group node)
    ACTOR = "ACTOR"               # the "whom" of a user-interaction requirement
    PROCESS = "PROCESS"           # the process verb (the doing word)
    OBJECT = "OBJECT"             # the object the process operates on
    DETAILS = "DETAILS"           # additional object details / qualifier
    CONSTRAINT = "CONSTRAINT"     # quality/performance constraint (within 2 s, ...)
    OPERATOR = "OPERATOR"         # logical connective AND / OR / NOT (group node)
    GLUE = "GLUE"                 # literal connective/separator text (round-trip only)


class Rel(str, Enum):
    """Typed relationship carried by an edge."""
    HAS_CONDITION = "HAS_CONDITION"
    HAS_SUBJECT = "HAS_SUBJECT"
    HAS_MODALITY = "HAS_MODALITY"
    HAS_ACTION = "HAS_ACTION"
    HAS_ACTOR = "HAS_ACTOR"
    HAS_PROCESS = "HAS_PROCESS"
    ACTS_ON = "ACTS_ON"            # ACTION -> OBJECT
    HAS_DETAILS = "HAS_DETAILS"
    HAS_CONSTRAINT = "HAS_CONSTRAINT"
    OPERAND = "OPERAND"            # OPERATOR -> operand
    NEXT = "NEXT"                  # ordered sequence of terminals (text order)


# IREB: the modal verb fixes the *legal obligation* (binding force) of the requirement.
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


# ---------------------------------------------------------------------------
# 2. Graph data structure (lightweight, dependency-free)
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    role: Role
    text: str = ""                       # surface text (terminals only)
    attrs: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.role in (
            Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
            Role.PROCESS, Role.OBJECT, Role.DETAILS, Role.CONSTRAINT, Role.GLUE,
        )


@dataclass
class Edge:
    source: str
    target: str
    rel: Rel
    attrs: dict = field(default_factory=dict)


class RequirementGraph:
    """A typed, ordered graph for one requirement.

    The list ``leaf_order`` is the lossless backbone: the ids of all terminal
    nodes in *exact text order*. Concatenating their ``text`` reproduces the
    original requirement.
    """

    def __init__(self, template_name: str = "", metadata: Optional[dict] = None):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.leaf_order: list[str] = []
        self.root_id: Optional[str] = None
        self.template_name = template_name
        self.metadata = metadata or {}     # IREB attributes: id, source, rationale...
        self._counter = 0

    # -- construction helpers -------------------------------------------------
    def new_id(self, role: Role) -> str:
        self._counter += 1
        return f"{role.value.lower()}_{self._counter}"

    def add_node(self, role: Role, text: str = "", **attrs) -> Node:
        node = Node(self.new_id(role), role, text, dict(attrs))
        self.nodes[node.id] = node
        return node

    def add_edge(self, source: Node | str, target: Node | str, rel: Rel, **attrs) -> Edge:
        s = source.id if isinstance(source, Node) else source
        t = target.id if isinstance(target, Node) else target
        edge = Edge(s, t, rel, dict(attrs))
        self.edges.append(edge)
        return edge

    # -- generation (graph -> text), guaranteed lossless ----------------------
    def generate(self) -> str:
        """Regenerate the requirement text from the graph (R3)."""
        return "".join(self.nodes[nid].text for nid in self.leaf_order)

    # -- queries --------------------------------------------------------------
    def elements(self) -> list[Node]:
        """Semantic (non-glue) terminal elements in text order."""
        return [self.nodes[i] for i in self.leaf_order
                if self.nodes[i].role is not Role.GLUE and self.nodes[i].text.strip()]

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "template": self.template_name,
            "metadata": self.metadata,
            "root": self.root_id,
            "leaf_order": self.leaf_order,
            "nodes": [
                {"id": n.id, "role": n.role.value, "text": n.text, "attrs": n.attrs}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "rel": e.rel.value, "attrs": e.attrs}
                for e in self.edges
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RequirementGraph":
        g = cls(d.get("template", ""), d.get("metadata", {}))
        for nd in d["nodes"]:
            n = Node(nd["id"], Role(nd["role"]), nd.get("text", ""), nd.get("attrs", {}))
            g.nodes[n.id] = n
        for ed in d["edges"]:
            g.edges.append(Edge(ed["source"], ed["target"], Rel(ed["rel"]), ed.get("attrs", {})))
        g.leaf_order = d["leaf_order"]
        g.root_id = d.get("root")
        return g

    # -- visual / interop exporters ------------------------------------------
    def to_mermaid(self, show_glue: bool = False) -> str:
        """Mermaid flowchart (paste into any Markdown / mermaid.live)."""
        lines = ["flowchart TD"]
        shapes = {
            Role.ROOT: ("([", "])"), Role.OPERATOR: ("{{", "}}"),
            Role.ACTION: ("[/", "/]"),
        }
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            label = n.role.value if not n.text.strip() else f"{n.role.value}: {n.text.strip()}"
            label = label.replace('"', "'")
            o, c = shapes.get(n.role, ("[", "]"))
            lines.append(f'    {n.id}{o}"{label}"{c}')
        for e in self.edges:
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and (self.nodes[e.target].role is Role.GLUE):
                continue
            lines.append(f"    {e.source} -->|{e.rel.value}| {e.target}")
        return "\n".join(lines)

    def to_dot(self, show_glue: bool = False) -> str:
        """Graphviz DOT."""
        lines = ["digraph Requirement {", "  rankdir=TB;", '  node [fontname="Helvetica"];']
        for n in self.nodes.values():
            if n.role is Role.GLUE and not show_glue:
                continue
            label = n.role.value if not n.text.strip() else f"{n.role.value}\\n{n.text.strip()}"
            label = label.replace('"', "'")
            lines.append(f'  "{n.id}" [label="{label}"];')
        for e in self.edges:
            if e.rel is Rel.NEXT:
                continue
            if not show_glue and self.nodes[e.target].role is Role.GLUE:
                continue
            lines.append(f'  "{e.source}" -> "{e.target}" [label="{e.rel.value}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_cypher(self) -> str:
        """Neo4j Cypher CREATE statements (graph-DB ingestion)."""
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
            stmts.append(
                f"MATCH (a {{id:'{e.source}'}}),(b {{id:'{e.target}'}}) "
                f"CREATE (a)-[:{e.rel.value}]->(b)"
            )
        return "\n".join(stmts)

    def to_networkx(self):
        """Return a networkx.DiGraph (only if networkx is installed)."""
        import networkx as nx  # optional dependency
        g = nx.DiGraph(template=self.template_name, **self.metadata)
        for n in self.nodes.values():
            g.add_node(n.id, role=n.role.value, text=n.text, **n.attrs)
        for e in self.edges:
            g.add_edge(e.source, e.target, rel=e.rel.value, **e.attrs)
        return g

    def summary(self) -> str:
        """Human-readable element table for a requirements engineer."""
        lines = []
        if self.metadata:
            lines.append("Attributes: " + ", ".join(f"{k}={v}" for k, v in self.metadata.items()))
        lines.append(f"Template  : {self.template_name}")
        lines.append("Elements  :")
        for n in self.elements():
            extra = ""
            if n.role is Role.MODALITY:
                extra = f"   [{n.attrs.get('obligation','')}]"
            elif n.attrs.get("function_type"):
                extra = f"   [function: {n.attrs['function_type']}]"
            elif n.attrs.get("kind"):
                extra = f"   [{n.attrs['kind']}]"
            lines.append(f"   - {n.role.value:<11}: {n.text.strip()!r}{extra}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Template = configuration of one requirement structure (R1, R4)
# ---------------------------------------------------------------------------

@dataclass
class Template:
    """A configurable requirement *structure*.

    Everything the parser needs to recognise the elements of a requirement
    style, and everything the renderer needs to lay them out again.
    """
    name: str
    # markers (lower-case); multi-word entries are fine ("as soon as")
    condition_markers: tuple = (
        "when", "while", "if", "as soon as", "once", "after", "before",
        "during", "in case of", "unless", "as long as", "provided that",
        "given that", "whenever",
    )
    modality_keywords: tuple = (
        "shall not", "should not", "must not", "will not", "may not",
        "shall", "should", "must", "will", "may", "cannot", "can",
    )
    # user-interaction (Rupp type 2): "provide <whom> with the ability to <verb>"
    user_interaction_open: tuple = ("provide",)
    user_interaction_bridge: tuple = ("with the ability to",)
    # interface (Rupp type 3): "be able to <verb>"
    interface_markers: tuple = ("be able to",)
    constraint_markers: tuple = (
        "within", "in less than", "in no more than", "no later than",
        "in at most", "at a rate of", "with an accuracy of",
        "with a resolution of", "with a precision of", "for at least",
        "for no longer than", "every",
    )
    object_determiners: tuple = (
        "the", "a", "an", "its", "their", "each", "every", "all", "any",
        "this", "that", "these", "those", "his", "her", "our", "your",
    )
    conjunctions: tuple = ("and", "or")
    # slot order used when *rendering* a brand-new requirement from elements
    slot_order: tuple = (
        Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
        Role.PROCESS, Role.OBJECT, Role.DETAILS, Role.CONSTRAINT,
    )

    def _alt(self, items) -> str:
        return "|".join(re.escape(w) for w in sorted(items, key=len, reverse=True))


# --- shipped templates ------------------------------------------------------

# IREB / Rupp "MASTeR" boilerplate (CPRE Foundation Level).
RUPP_TEMPLATE = Template(name="IREB-Rupp")

# EARS - Easy Approach to Requirements Syntax (very common in aerospace).
#   Ubiquitous : "The <system> shall <response>"
#   Event      : "WHEN <trigger> the <system> shall <response>"
#   State      : "WHILE <state> the <system> shall <response>"
#   Unwanted   : "IF <condition> THEN the <system> shall <response>"
#   Optional   : "WHERE <feature> the <system> shall <response>"
EARS_TEMPLATE = Template(
    name="EARS",
    condition_markers=("when", "while", "if", "where", "as soon as"),
    # EARS keeps the action as a free response; no Rupp type-2/3 phrasing
    user_interaction_open=(),
    interface_markers=(),
    slot_order=(Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.PROCESS,
                Role.OBJECT, Role.CONSTRAINT),
)

TEMPLATES = {t.name: t for t in (RUPP_TEMPLATE, EARS_TEMPLATE)}


def register_template(template: Template) -> None:
    """Register a custom requirement structure so it can be used by name (R4)."""
    TEMPLATES[template.name] = template


# ---------------------------------------------------------------------------
# 4. Parser : text -> RequirementGraph  (R1, R2, R3)
# ---------------------------------------------------------------------------

class RequirementParser:
    """Anchor-based parser. Recognises template elements and *tiles* the text."""

    def __init__(self, template: Template = RUPP_TEMPLATE):
        self.t = template

    # span = (start, end, Role, attrs)
    def _claims_to_graph(self, text: str, claims: list, metadata: dict,
                         action_struct: dict) -> RequirementGraph:
        g = RequirementGraph(self.t.name, metadata)

        # resolve overlaps: keep earliest, drop anything overlapping it
        claims = sorted(claims, key=lambda c: c[0])
        accepted = []
        last_end = -1
        for c in claims:
            if c[0] >= last_end and c[1] > c[0]:
                accepted.append(c)
                last_end = c[1]

        # tile: fill gaps with GLUE so concatenation == original (lossless)
        terminals: list[Node] = []
        span_node: dict[tuple, Node] = {}
        pos = 0
        for (s, e, role, attrs) in accepted:
            if s > pos:
                terminals.append(g.add_node(Role.GLUE, text[pos:s]))
            node = g.add_node(role, text[s:e], **attrs)
            span_node[(s, e)] = node
            terminals.append(node)
            pos = e
        if pos < len(text):
            terminals.append(g.add_node(Role.GLUE, text[pos:]))

        g.leaf_order = [n.id for n in terminals]
        # ordered NEXT chain (handy for path queries / re-serialisation)
        for a, b in zip(terminals, terminals[1:]):
            g.add_edge(a, b, Rel.NEXT)

        # ---- build the semantic tree on top of the terminals ----
        root = g.add_node(Role.ROOT, **metadata)
        g.root_id = root.id

        def node_for(role):
            for (s, e, r, _a) in accepted:
                if r is role:
                    return span_node[(s, e)]
            return None

        cond = [span_node[(s, e)] for (s, e, r, _a) in accepted if r is Role.CONDITION]
        for c in cond:
            g.add_edge(root, c, Rel.HAS_CONDITION)
        subj = node_for(Role.SUBJECT)
        if subj:
            g.add_edge(root, subj, Rel.HAS_SUBJECT)
        modal = node_for(Role.MODALITY)
        if modal:
            g.add_edge(root, modal, Rel.HAS_MODALITY)
        actor = node_for(Role.ACTOR)
        constraint = node_for(Role.CONSTRAINT)

        # action(s): one ACTION group per (process,object) pair, joined by an
        # OPERATOR node if the requirement contains a compound action.
        action_groups = []
        for (proc_span, obj_span) in action_struct["pairs"]:
            act = g.add_node(Role.ACTION,
                             function_type=action_struct["function_type"])
            if proc_span and proc_span in span_node:
                g.add_edge(act, span_node[proc_span], Rel.HAS_PROCESS)
            if obj_span and obj_span in span_node:
                g.add_edge(act, span_node[obj_span], Rel.ACTS_ON)
            action_groups.append(act)

        if action_groups:
            if len(action_groups) == 1:
                g.add_edge(root, action_groups[0], Rel.HAS_ACTION)
                head = action_groups[0]
            else:
                op = g.add_node(Role.OPERATOR, text="",
                                operator=action_struct["operator"])
                g.add_edge(root, op, Rel.HAS_ACTION)
                for act in action_groups:
                    g.add_edge(op, act, Rel.OPERAND)
                head = action_groups[0]
            if actor:
                g.add_edge(head, actor, Rel.HAS_ACTOR)
            if constraint:
                g.add_edge(head, constraint, Rel.HAS_CONSTRAINT)
        elif constraint:
            g.add_edge(root, constraint, Rel.HAS_CONSTRAINT)

        return g

    # --- helpers -------------------------------------------------------------
    def _search(self, pattern, text, start, end):
        m = re.compile(pattern, re.I).search(text, start, end)
        return m

    def _first_determiner(self, text, start, end):
        if not self.t.object_determiners:
            return None
        pat = r"(?<![\w'])(" + self.t._alt(self.t.object_determiners) + r")(?![\w'])"
        m = self._search(pat, text, start, end)
        return m.start() if m else None

    @staticmethod
    def _trim(text, s, e):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e

    def _parse_action(self, text, a_start, a_end):
        """Split a compound action and extract (process, object) pairs.

        Returns claims (spans) and a structure dict describing the grouping.
        """
        a_start, a_end = self._trim(text, a_start, a_end)
        claims = []
        pairs = []
        operator = None

        # detect a single top-level conjunction that introduces a *new verb*
        # (i.e. the part after it does NOT start with a determiner -> it is a
        # second action, not a compound object).
        split_at = None
        if self.t.conjunctions:
            conj_pat = r"\s+(" + self.t._alt(self.t.conjunctions) + r")\s+"
            for m in re.finditer(conj_pat, text[a_start:a_end], re.I):
                after = a_start + m.end()
                nxt = re.match(r"\s*([\w']+)", text[after:a_end])
                first_word = nxt.group(1).lower() if nxt else ""
                # treat as action split only if the next word is not a
                # determiner (heuristic: a new verb phrase, not a compound object)
                if first_word and first_word not in self.t.object_determiners:
                    split_at = (a_start + m.start(), after, m.group(1).lower())
                    break

        def simple(seg_s, seg_e):
            seg_s, seg_e = self._trim(text, seg_s, seg_e)
            d = self._first_determiner(text, seg_s, seg_e)
            if d is not None and d > seg_s:
                ps, pe = self._trim(text, seg_s, d)
                os, oe = self._trim(text, d, seg_e)
                proc = (ps, pe) if pe > ps else None
                obj = (os, oe) if oe > os else None
            else:
                proc = (seg_s, seg_e) if seg_e > seg_s else None
                obj = None
            if proc:
                claims.append((proc[0], proc[1], Role.PROCESS, {}))
            if obj:
                claims.append((obj[0], obj[1], Role.OBJECT, {}))
            pairs.append((proc, obj))

        if split_at:
            operator = split_at[2].upper()
            simple(a_start, split_at[0])
            simple(split_at[1], a_end)
        else:
            simple(a_start, a_end)

        return claims, {"pairs": pairs, "operator": operator}

    # --- main entry ----------------------------------------------------------
    def parse(self, text: str, metadata: Optional[dict] = None) -> RequirementGraph:
        metadata = dict(metadata or {})
        claims = []
        n = len(text)
        work_start = 0

        # (1) detect a leading condition marker and find the modality. We locate
        # the modality first so the condition can end at the *last comma before
        # the modality* -- this avoids splitting on commas inside the condition
        # itself (e.g. the thousands separator in "14,000 feet").
        cond_pat = r"^(\s*)(" + self.t._alt(self.t.condition_markers) + r")(?![\w'])"
        mcond = re.compile(cond_pat, re.I).match(text)
        first_comma = text.find(",", mcond.end()) if mcond else -1
        mod_search_start = (first_comma + 1 if first_comma != -1 else mcond.end()) \
            if mcond else 0

        mod_pat = r"(?<![\w'])(" + self.t._alt(self.t.modality_keywords) + r")(?![\w'])"
        mm = self._search(mod_pat, text, mod_search_start, n)
        mod_start = mm.start(1) if mm else n

        if mcond:
            last_comma = text.rfind(",", mcond.end(), mod_start)
            stop = last_comma if last_comma != -1 else first_comma
            if stop == -1:  # comma-less conditional, terminated by "then"
                tn = re.compile(r"(?<![\w'])then(?![\w'])", re.I).search(text, mcond.end(), mod_start)
                stop = tn.start() if tn else -1
            if stop != -1:
                cs, ce = self._trim(text, mcond.start(2), stop)
                claims.append((cs, ce, Role.CONDITION,
                               {"marker": mcond.group(2).lower(), "kind": "pre-condition"}))
                work_start = stop + 1 if text[stop:stop + 1] == "," else stop
                # drop a leading EARS-style "then" connector from the subject
                tm = re.compile(r"\s*then(?![\w'])", re.I).match(text, work_start)
                if tm:
                    work_start = tm.end()

        # (2) modality (anchor): everything before it (after condition) = subject
        post_start = n
        if mm:
            kw = mm.group(1).lower()
            claims.append((mm.start(1), mm.end(1), Role.MODALITY,
                           {"obligation": OBLIGATION.get(kw, "unspecified")}))
            ss, se = self._trim(text, work_start, mm.start(1))
            if se > ss:
                claims.append((ss, se, Role.SUBJECT, {}))
            post_start = mm.end(1)
        else:
            # no modal verb: treat the whole tail as subject-less action
            post_start = work_start

        post_end = n
        # strip a single trailing sentence terminator into glue
        while post_end > post_start and text[post_end - 1] in ".;\n\t ":
            post_end -= 1

        # (3) trailing condition: "<marker> ..." after the modality
        tc = self._search(r"(?<![\w'])(" + self.t._alt(self.t.condition_markers) + r")(?![\w'])",
                          text, post_start, post_end)
        if tc and tc.start() > post_start:
            cs, ce = self._trim(text, tc.start(), post_end)
            claims.append((cs, ce, Role.CONDITION,
                           {"marker": tc.group(1).lower(), "kind": "post-condition"}))
            post_end = tc.start()

        # (4) trailing constraint (performance / quality)
        if self.t.constraint_markers:
            cc = self._search(r"(?<![\w'])(" + self.t._alt(self.t.constraint_markers) + r")(?![\w'])",
                              text, post_start, post_end)
            if cc and cc.start() > post_start:
                cs, ce = self._trim(text, cc.start(), post_end)
                claims.append((cs, ce, Role.CONSTRAINT,
                               {"marker": cc.group(1).lower()}))
                post_end = cc.start()

        # (5) function type (Rupp): user-interaction / interface / autonomous
        function_type = "autonomous activity"
        action_start = post_start
        if self.t.user_interaction_open and self.t.user_interaction_bridge:
            ui = self._search(
                r"(?<![\w'])(" + self.t._alt(self.t.user_interaction_open) + r")\s+(.+?)\s+(" +
                self.t._alt(self.t.user_interaction_bridge) + r")(?![\w'])",
                text, post_start, post_end)
            if ui:
                function_type = "user interaction"
                asx, aex = self._trim(text, ui.start(2), ui.end(2))
                claims.append((asx, aex, Role.ACTOR, {}))
                action_start = ui.end(3)
        if function_type == "autonomous activity" and self.t.interface_markers:
            itf = self._search(r"(?<![\w'])(" + self.t._alt(self.t.interface_markers) + r")(?![\w'])",
                               text, post_start, post_end)
            if itf:
                function_type = "interface requirement"
                action_start = itf.end(1)

        # (6) the action core -> process / object (+ compound operator)
        act_claims, act_struct = self._parse_action(text, action_start, post_end)
        act_struct["function_type"] = function_type
        claims.extend(act_claims)

        return self._claims_to_graph(text, claims, metadata, act_struct)


# ---------------------------------------------------------------------------
# 5. Renderer : build a NEW requirement (graph) from elements (R3, R4)
# ---------------------------------------------------------------------------

def build_requirement(elements: dict, template: Template = RUPP_TEMPLATE,
                      metadata: Optional[dict] = None) -> RequirementGraph:
    """Create a requirement graph from scratch out of semantic elements.

    ``elements`` keys are Role members (or their names). Example::

        build_requirement({
            Role.CONDITION: "When the cabin altitude exceeds 14,000 ft",
            Role.SUBJECT:   "the oxygen system",
            Role.MODALITY:  "shall",
            Role.PROCESS:   "deploy",
            Role.OBJECT:    "the passenger oxygen masks",
            Role.CONSTRAINT:"within 4 seconds",
        })

    The text is rendered in the template's slot order, then re-parsed so the
    returned graph is structurally identical to a parsed one (and round-trips).
    """
    norm = {}
    for k, v in elements.items():
        role = k if isinstance(k, Role) else Role[str(k).upper()]
        norm[role] = v.strip()

    parts = []
    for role in template.slot_order:
        if role in norm and norm[role]:
            val = norm[role]
            if role is Role.CONDITION and not val.endswith(","):
                val = val + ","
            parts.append(val)
    text = " ".join(parts)
    if not text.endswith((".", "!", "?")):
        text += "."
    return RequirementParser(template).parse(text, metadata)


# ---------------------------------------------------------------------------
# 6. Convenience round-trip verifier (R3)
# ---------------------------------------------------------------------------

def roundtrip_ok(text: str, template: Template = RUPP_TEMPLATE) -> bool:
    return RequirementParser(template).parse(text).generate() == text


# ---------------------------------------------------------------------------
# 7. Demonstration
# ---------------------------------------------------------------------------

def _demo():
    bar = "=" * 78

    print(bar)
    print("DEMO 1 - IREB/Rupp boilerplate, increasing complexity, lossless round-trip")
    print(bar)

    requirements = [
        # autonomous activity
        ("REQ-001",
         "The flight management system shall calculate the optimal cruise altitude."),
        # event condition + performance constraint
        ("REQ-002",
         "When the cabin altitude exceeds 14,000 feet, the oxygen system shall "
         "deploy the passenger oxygen masks within 4 seconds."),
        # user-interaction (Rupp type 2) + post-condition
        ("REQ-003",
         "While the aircraft is on the ground, the avionics suite shall provide "
         "the pilot with the ability to configure the flight plan if the parking "
         "brake is set."),
        # compound action (logical AND) + condition + constraint  (the complex one)
        ("REQ-004",
         "As soon as an engine fire is detected, the engine control unit shall "
         "shut off the affected engine and activate the fire suppression system "
         "within 500 milliseconds."),
    ]

    parser = RequirementParser(RUPP_TEMPLATE)
    for rid, text in requirements:
        g = parser.parse(text, metadata={"id": rid, "verification": "Test"})
        regenerated = g.generate()
        print(f"\n[{rid}] {text}")
        print(g.summary())
        print(f"   round-trip exact : {regenerated == text}")
        assert regenerated == text, "LOSSLESS ROUND-TRIP FAILED"

    print("\n" + bar)
    print("DEMO 2 - Graph of the complex requirement REQ-004 (Mermaid)")
    print(bar)
    g4 = parser.parse(requirements[3][1], metadata={"id": "REQ-004"})
    print(g4.to_mermaid())

    print("\n" + bar)
    print("DEMO 3 - Custom structure: EARS template (aerospace)")
    print(bar)
    ears = RequirementParser(EARS_TEMPLATE)
    ears_text = ("If the airspeed drops below the stall speed, then the "
                 "stall warning system shall activate the stick shaker.")
    ge = ears.parse(ears_text, metadata={"id": "REQ-EARS-1"})
    print(f"\n{ears_text}")
    print(ge.summary())
    print(f"   round-trip exact : {ge.generate() == ears_text}")

    print("\n" + bar)
    print("DEMO 4 - Register your OWN custom requirement structure")
    print(bar)
    # A minimalist 'CONTRACT' style: <subject> is required to <process> <object>
    contract = Template(
        name="CONTRACT",
        condition_markers=("when", "if"),
        modality_keywords=("is required to", "is permitted to", "is prohibited from"),
        user_interaction_open=(), interface_markers=(),
        slot_order=(Role.CONDITION, Role.SUBJECT, Role.MODALITY,
                    Role.PROCESS, Role.OBJECT, Role.CONSTRAINT),
    )
    register_template(contract)
    ctext = "The ground station is required to acknowledge the telemetry packet within 2 seconds."
    gc = RequirementParser(contract).parse(ctext, {"id": "REQ-CT-1"})
    print(f"\n{ctext}")
    print(gc.summary())
    print(f"   round-trip exact : {gc.generate() == ctext}")

    print("\n" + bar)
    print("DEMO 5 - Build a NEW requirement from semantic elements, then emit it")
    print(bar)
    built = build_requirement({
        Role.CONDITION: "When the landing gear is selected down",
        Role.SUBJECT:   "the landing gear control system",
        Role.MODALITY:  "shall",
        Role.PROCESS:   "extend",
        Role.OBJECT:    "the main landing gear",
        Role.CONSTRAINT: "within 10 seconds",
    }, metadata={"id": "REQ-LG-1", "source": "ATA-32", "rationale": "certification CS-25"})
    print("\nGenerated requirement text:")
    print("   " + built.generate())
    print(built.summary())

    print("\n" + bar)
    print("DEMO 6 - JSON serialisation round-trip (persist / load the graph)")
    print(bar)
    js = g4.to_json()
    reloaded = RequirementGraph.from_dict(json.loads(js))
    print(f"   JSON length         : {len(js)} chars")
    print(f"   reload regenerates  : {reloaded.generate() == requirements[3][1]}")

    print("\nAll lossless round-trip checks passed.")


if __name__ == "__main__":
    _demo()
