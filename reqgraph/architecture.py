"""
reqgraph.architecture
=====================

A structural view of a system architecture for the architecture review agents
(:mod:`reqgraph.agents.architecture`): what the elements are, how they are
composed, which requirements the model itself contains, and the explicit trace
links between them.

:mod:`reqgraph.sysml_parser` extracts element *names* for semantic matching;
it does not keep composition, requirement identifiers or trace relationships.
This module does, for SysML v2 textual notation:

* definitions and usages (``part def`` / ``part p : T`` …) with their ``doc``
  text and owner, giving a composition tree (system → subsystem → component);
* requirements (``requirement [def] <'REQ-1'> Name { doc … subject s; }``)
  including requirements nested inside requirements (sub/derived);
* trace links: ``satisfy [requirement] R by p;`` (or ``satisfy R;`` inside a
  part), ``allocate a to b;``, ``verify [requirement] R;`` inside a
  verification, requirement ``subject`` bindings, ``dependency from a to b;``
  and ``#derivation`` connections (``end #original ::> A; end #derive ::> B;``).

SysML v1 models (XMI, Turtle, Cameo ``.mdzip``) are adapted from
:class:`~reqgraph.sysml_v1_parser.SysMLV1Model`, so the agents work on either.

The extractor is regex-based (like the rest of RAVEN's SysML support): it reads
the common textual constructs, not the full KerML grammar. Anything it cannot
place is simply absent from the model; it never invents links.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .errors import DataFormatError


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

# element kinds that represent structure an architect would expect a
# requirement to drive (gold-plating candidates when nothing does)
STRUCTURAL_KINDS = {"part_def", "port_def", "interface_def", "connection_def",
                    "item_def", "action_def", "state_def"}


@dataclass
class ArchElement:
    """A definition (``part def X``) or usage (``part x : X``) in the model."""
    name: str
    kind: str                    # part_def, part, port_def, port, action_def, …
    doc: str = ""
    owner: str = ""              # enclosing definition / usage name
    type: str = ""               # for usages: the definition it is typed by
    package: str = ""
    line: int = 0


@dataclass
class ArchRequirement:
    """A requirement written inside the architecture model."""
    name: str
    short_id: str = ""           # requirement <'REQ-1'> Name
    text: str = ""               # doc text (or text "…")
    subjects: list = field(default_factory=list)
    owner: str = ""              # enclosing requirement, for nested ones
    is_def: bool = False
    line: int = 0


@dataclass
class ArchLink:
    """An explicit trace relationship stated in the model."""
    kind: str                    # satisfy | allocate | verify | subject | derive | dependency
    source: str                  # satisfier / allocated / verifier / derived element
    target: str                  # requirement / allocation target / original
    line: int = 0


@dataclass
class ArchitectureModel:
    elements: list = field(default_factory=list)       # [ArchElement]
    requirements: list = field(default_factory=list)   # [ArchRequirement]
    links: list = field(default_factory=list)          # [ArchLink]
    source_path: str = ""
    notation: str = "sysml-v2"

    # -- lookup -------------------------------------------------------------
    def definitions(self) -> list:
        return [e for e in self.elements if e.kind.endswith("_def")]

    def _def_index(self) -> dict:
        return {e.name.lower(): e for e in self.definitions()}

    def resolve(self, ref: str) -> Optional[str]:
        """Map a reference (usage name, qualified path, def name) to a
        definition name, or None when it does not resolve."""
        if not ref:
            return None
        last = re.split(r"::|\.", ref.strip())[-1].strip("'\" ")
        defs = self._def_index()
        if last.lower() in defs:
            return defs[last.lower()].name
        for e in self.elements:
            if not e.kind.endswith("_def") and e.name.lower() == last.lower() and e.type:
                t = re.split(r"::|\.", e.type)[-1]
                if t.lower() in defs:
                    return defs[t.lower()].name
        squashed = re.sub(r"[^a-z0-9]", "", last.lower())
        for name, d in defs.items():
            if re.sub(r"[^a-z0-9]", "", name) == squashed:
                return d.name
        return None

    def requirement(self, ref: str) -> Optional[ArchRequirement]:
        last = re.split(r"::|\.", (ref or "").strip())[-1].strip("'\" ")
        for r in self.requirements:
            if last and (r.name.lower() == last.lower() or r.short_id.lower() == last.lower()):
                return r
        return None

    # -- composition ---------------------------------------------------------
    def children(self, def_name: str) -> list:
        """Definitions directly composed by ``def_name`` (via typed usages)."""
        out = []
        for e in self.elements:
            if e.owner == def_name and not e.kind.endswith("_def") and e.type:
                d = self.resolve(e.type)
                if d and d != def_name and d not in out:
                    out.append(d)
        return out

    def parents(self, def_name: str) -> list:
        return [d.name for d in self.definitions() if def_name in self.children(d.name)]

    def depth(self) -> dict:
        """Composition depth of each definition (roots = 0)."""
        defs = [d.name for d in self.definitions() if d.kind in ("part_def",)]
        composed = {c for d in defs for c in self.children(d)}
        roots = [d for d in defs if d not in composed] or defs[:1]
        depth, frontier = {}, [(r, 0) for r in roots]
        while frontier:
            name, d = frontier.pop(0)
            if name in depth and depth[name] <= d:
                continue
            depth[name] = d
            frontier += [(c, d + 1) for c in self.children(name)]
        return depth

    def descendants(self, def_name: str) -> set:
        seen, stack = set(), [def_name]
        while stack:
            for c in self.children(stack.pop()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    # -- serialisation (agent state must be JSON-friendly) -------------------
    def to_dict(self) -> dict:
        return {"notation": self.notation, "source_path": self.source_path,
                "elements": [asdict(e) for e in self.elements],
                "requirements": [asdict(r) for r in self.requirements],
                "links": [asdict(l) for l in self.links]}

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureModel":
        return cls(elements=[ArchElement(**e) for e in d.get("elements", [])],
                   requirements=[ArchRequirement(**r) for r in d.get("requirements", [])],
                   links=[ArchLink(**l) for l in d.get("links", [])],
                   source_path=d.get("source_path", ""),
                   notation=d.get("notation", "sysml-v2"))

    def summary(self) -> dict:
        kinds = {}
        for e in self.elements:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        links = {}
        for l in self.links:
            links[l.kind] = links.get(l.kind, 0) + 1
        return {"notation": self.notation, "elements": len(self.elements),
                "by_kind": kinds, "requirements": len(self.requirements),
                "links": links}


# ---------------------------------------------------------------------------
# SysML v2 textual notation
# ---------------------------------------------------------------------------

_DOC_RE = re.compile(r"\bdoc\s*/\*(.*?)\*/", re.S)
_KINDS = (r"part|action|state|attribute|port|connection|interface|item|requirement|"
          r"constraint|use\s+case|analysis|verification|occurrence|allocation|calc|"
          r"concern|enum|flow")
_PREFIX = r"(?:(?:abstract|ref|in|out|inout|private|public|protected|derived|readonly|variation|individual)\s+)*"
_NAME = r"(?:'[^']+'|[A-Za-z_]\w*)"
_SCAN_RE = re.compile(
    r"(?P<doc>\bdoc\b)"
    r"|(?P<satisfy>\b(?:not\s+)?satisfy\s+(?:requirement\s+)?(?P<sat_req>[\w:.']+)(?:\s*:\s*[\w:.]+)?"
    r"(?:\s+by\s+(?P<sat_by>[\w:.']+))?\s*[;{])"
    r"|(?P<allocate>\ballocate\s+(?P<alloc_src>[\w:.']+)\s+to\s+(?P<alloc_tgt>[\w:.']+)\s*[;{])"
    r"|(?P<verify>\bverify\s+(?:requirement\s+)?(?P<ver_req>[\w:.']+)(?:\s*:\s*[\w:.]+)?\s*[;{])"
    r"|(?P<subject>\bsubject\s+(?P<subj>[\w']+)(?:\s*:\s*(?P<subj_type>[\w:.]+))?\s*[;={])"
    r"|(?P<dependency>\bdependency\s+(?:\w+\s+)?from\s+(?P<dep_src>[\w:.']+)\s+to\s+(?P<dep_tgt>[\w:.']+)\s*[;{])"
    r"|(?P<end>\bend\s+#(?P<end_tag>original|derive|derived)\s*(?:::>|:>|:)\s*(?P<end_ref>[\w:.']+)\s*[;{])"
    r"|(?P<pkg>\bpackage\s+(?P<pkg_name>" + _NAME + r")\s*\{)"
    r"|(?P<decl>(?P<deriv>#derivation\s+)?" + _PREFIX + r"\b(?P<kw>" + _KINDS + r")(?P<isdef>\s+def)?"
    r"\s+(?:<'?(?P<sid>[^'>]+)'?>\s*)?(?P<name>" + _NAME + r")?"
    r"(?P<rest>[^{};]*?)(?P<term>[{;]))"
    r"|(?P<open>\{)|(?P<close>\})",
)
_TYPED_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?::|:>|::>)\s*([\w:.']+)")
_TEXT_FIELD_RE = re.compile(r'\btext\s*=?\s*"([^"]*)"')


def _clean_name(n: str) -> str:
    return (n or "").strip().strip("'")


def parse_architecture_sysml(text: str, source_path: str = "") -> ArchitectureModel:
    """Extract an :class:`ArchitectureModel` from SysML v2 textual notation."""
    if not text or not text.strip():
        raise DataFormatError("architecture model is empty")
    docs = {m.start(): " ".join(m.group(1).split()) for m in _DOC_RE.finditer(text)}

    def blank(m):
        return " " * len(m.group())
    clean = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    clean = re.sub(r"//[^\n]*", blank, clean)

    model = ArchitectureModel(source_path=source_path)
    frames: list = []          # [{"kind","name","pkg","obj"}]
    pkgs: list = []
    derivation: Optional[dict] = None

    def owner_frame(kinds=None):
        for f in reversed(frames):
            if f["name"] and (kinds is None or f["kind"] in kinds):
                return f
        return None

    def line_of(pos):
        return text.count("\n", 0, pos) + 1

    for m in _SCAN_RE.finditer(clean):
        pos = m.start()
        if m.group("doc"):
            body = docs.get(pos)
            f = frames[-1] if frames else None
            if body and f and f.get("obj") is not None and not f["obj"].__dict__.get("_doc_set"):
                obj = f["obj"]
                if isinstance(obj, ArchRequirement):
                    obj.text = obj.text or body
                else:
                    obj.doc = obj.doc or body
                obj.__dict__["_doc_set"] = True
            continue
        if m.group("pkg"):
            pkgs.append(_clean_name(m.group("pkg_name")))
            frames.append({"kind": "package", "name": pkgs[-1], "obj": None})
            continue
        if m.group("satisfy"):
            req = _clean_name(m.group("sat_req"))
            by = _clean_name(m.group("sat_by") or "")
            if not by:
                f = owner_frame({"part", "part_def", "action", "action_def", "item_def",
                                 "port_def", "interface_def"})
                by = f["name"] if f else ""
            if by:
                model.links.append(ArchLink("satisfy", by, req, line_of(pos)))
            if m.group(0).rstrip().endswith("{"):
                frames.append({"kind": "satisfy", "name": "", "obj": None})
            continue
        if m.group("allocate"):
            model.links.append(ArchLink("allocate", _clean_name(m.group("alloc_src")),
                                        _clean_name(m.group("alloc_tgt")), line_of(pos)))
            if m.group(0).rstrip().endswith("{"):
                frames.append({"kind": "allocate", "name": "", "obj": None})
            continue
        if m.group("verify"):
            f = owner_frame({"verification", "verification_def", "analysis", "analysis_def"})
            model.links.append(ArchLink("verify", f["name"] if f else "", _clean_name(m.group("ver_req")),
                                        line_of(pos)))
            if m.group(0).rstrip().endswith("{"):
                frames.append({"kind": "verify", "name": "", "obj": None})
            continue
        if m.group("subject"):
            f = owner_frame({"requirement", "requirement_def", "concern", "concern_def",
                             "use_case", "use_case_def"})
            if f and isinstance(f.get("obj"), ArchRequirement):
                subj = _clean_name(m.group("subj_type") or m.group("subj"))
                f["obj"].subjects.append(subj)
                model.links.append(ArchLink("subject", subj, f["obj"].name, line_of(pos)))
            if m.group(0).rstrip().endswith("{"):
                frames.append({"kind": "subject", "name": "", "obj": None})
            continue
        if m.group("dependency"):
            model.links.append(ArchLink("dependency", _clean_name(m.group("dep_src")),
                                        _clean_name(m.group("dep_tgt")), line_of(pos)))
            continue
        if m.group("end"):
            if derivation is not None:
                derivation["derived" if m.group("end_tag").startswith("deriv") else "original"] \
                    .append(_clean_name(m.group("end_ref")))
            continue
        if m.group("decl"):
            kw = re.sub(r"\s+", "_", m.group("kw").lower())
            kind = kw + ("_def" if m.group("isdef") else "")
            name = _clean_name(m.group("name") or "")
            opens = m.group("term") == "{"
            obj = None
            outer = owner_frame()
            if m.group("deriv") and opens:
                derivation = {"original": [], "derived": [], "depth": len(frames) + 1}
            if kw == "requirement" and name:
                outer_req = owner_frame({"requirement", "requirement_def"})
                obj = ArchRequirement(name=name, short_id=(m.group("sid") or "").strip(),
                                      owner=outer_req["name"] if outer_req else "",
                                      is_def=bool(m.group("isdef")), line=line_of(pos))
                tf = _TEXT_FIELD_RE.search(m.group("rest") or "")
                if tf:
                    obj.text = tf.group(1)
                model.requirements.append(obj)
                if not obj.is_def:
                    typed = _TYPED_RE.match(m.group("rest") or "")
                    if typed and outer and outer["kind"] in ("part_def", "part"):
                        model.links.append(ArchLink("satisfy", outer["name"], obj.name, line_of(pos)))
            elif name and kw not in ("attribute", "enum", "constraint", "flow"):
                typed = _TYPED_RE.match(m.group("rest") or "")
                obj = ArchElement(name=name, kind=kind,
                                  owner=outer["name"] if outer and outer["kind"] != "package" else "",
                                  type=_clean_name(typed.group(1)) if typed and not m.group("isdef") else "",
                                  package=pkgs[-1] if pkgs else "", line=line_of(pos))
                model.elements.append(obj)
            if opens:
                frames.append({"kind": kind, "name": name, "obj": obj})
            continue
        if m.group("open"):
            frames.append({"kind": "block", "name": "", "obj": None})
            continue
        if m.group("close"):
            if frames:
                closed = frames.pop()
                if derivation is not None and len(frames) + 1 == derivation["depth"]:
                    for o in derivation["original"]:
                        for d in derivation["derived"]:
                            model.links.append(ArchLink("derive", d, o, 0))
                    derivation = None
                if closed["kind"] == "package" and pkgs:
                    pkgs.pop()
            continue

    for obj in model.elements + model.requirements:
        obj.__dict__.pop("_doc_set", None)
        if isinstance(obj, ArchRequirement):
            obj.text = (obj.text or "").strip()
    if not model.elements and not model.requirements:
        raise DataFormatError(
            "no SysML v2 definitions or requirements found -- expected textual "
            "notation such as 'part def', 'requirement', 'satisfy … by …'")
    return model


# ---------------------------------------------------------------------------
# SysML v1 (XMI / Turtle / .mdzip) adapter
# ---------------------------------------------------------------------------

_V1_KIND = {"block": "part_def", "constraintblock": "part_def", "class": "part_def",
            "activity": "action_def", "action": "action_def", "statemachine": "state_def",
            "state": "state_def", "port": "port", "flowport": "port", "proxyport": "port",
            "fullport": "port", "interfaceblock": "interface_def", "interface": "interface_def",
            "property": "part"}


def from_sysml_v1(v1) -> ArchitectureModel:
    """Adapt a :class:`~reqgraph.sysml_v1_parser.SysMLV1Model`."""
    model = ArchitectureModel(source_path=getattr(v1, "source_path", ""), notation="sysml-v1")
    names = {}
    for e in v1.elements:
        names[e.xmi_id] = e.name
        st = (e.stereotype or "").lower()
        if "requirement" in st or e.req_id:
            model.requirements.append(ArchRequirement(
                name=e.name or e.req_id, short_id=e.req_id or "",
                text=e.req_text or e.doc or "", line=0))
            continue
        local = (e.stereotype or e.element_type.split(":")[-1]).replace(" ", "").lower()
        kind = _V1_KIND.get(local, "part_def" if "class" in e.element_type.lower() else "")
        if kind:
            model.elements.append(ArchElement(name=e.name, kind=kind, doc=e.doc or "",
                                              package=e.package or ""))
    by_name = {e.name: e for e in model.elements}
    for r in v1.relations:
        src, tgt = names.get(r.source_id, r.source_id), names.get(r.target_id, r.target_id)
        if r.rel_type == "composition":
            child = by_name.get(tgt)
            usage = ArchElement(name=r.name or tgt, kind="part", owner=src,
                                type=tgt if child else "")
            model.elements.append(usage)
        elif r.rel_type in ("satisfy", "allocate", "derive", "refine", "trace", "verify"):
            kind = "derive" if r.rel_type in ("derive", "refine") else r.rel_type
            model.links.append(ArchLink(kind, src, tgt, 0))
    return model


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_V1_EXTS = (".xmi", ".uml", ".ttl", ".rdf", ".owl", ".n3", ".mdzip", ".zip")


def load_architecture(source: str, *, fmt: Optional[str] = None,
                      source_path: str = "") -> ArchitectureModel:
    """Load an architecture from a file path or from model text.

    ``fmt`` may be ``"sysml-v2"`` or ``"sysml-v1"``; otherwise it is inferred
    from the extension or the content (XMI / Turtle → v1, else v2).
    """
    path = source if (len(source) < 4096 and "\n" not in source
                      and os.path.isfile(source)) else None
    if path:
        source_path = source_path or path
        ext = os.path.splitext(path)[1].lower()
        if fmt == "sysml-v1" or (fmt is None and ext in _V1_EXTS) or (
                fmt is None and ext == ".xml"):
            from .sysml_v1_parser import read_sysml_v1
            return from_sysml_v1(read_sysml_v1(path))
        with open(path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    head = source.lstrip()[:400]
    if fmt == "sysml-v1" or (fmt is None and (head.startswith("<?xml") or "<xmi:" in head
                                               or "<uml:" in head or head.startswith("@prefix"))):
        from .sysml_v1_parser import parse_sysml_v1
        return from_sysml_v1(parse_sysml_v1(source, source_path=source_path))
    return parse_architecture_sysml(source, source_path)
