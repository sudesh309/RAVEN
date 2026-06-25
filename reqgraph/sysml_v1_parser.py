"""
reqgraph.sysml_v1_parser
=========================

Parse SysML v1 XMI (MagicDraw/Cameo/Papyrus) and Turtle/RDF ontology exports
into a unified in-memory knowledge graph (``SysMLV1Model``).

Two input formats are supported and auto-detected:

* **XMI** (``.xmi``, ``.uml``, ``.xml``) — standard UML2 XMI with SysML
  stereotype applications as root-level siblings.  Tested against Cameo 18.x
  through 2022+ and Papyrus exports.
* **Turtle/RDF** (``.ttl``, ``.rdf``, ``.n3``, ``.owl``) — OWL/RDF ontology
  exported from a SysML tool (e.g. Cameo OWL exporter).  Requires
  ``rdflib>=6.0``.

Both paths produce the same ``SysMLV1Model`` so all downstream code
(comparison, KG export) is format-agnostic.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .core import Role
from .errors import DataFormatError

# ---------------------------------------------------------------------------
# Namespace / stereotype helpers
# ---------------------------------------------------------------------------

# Known SysML namespace URIs — used for stereotype-tag detection regardless
# of which XML prefix the file assigns to them.
_SYSML_PREFIXES = frozenset({
    "http://www.eclipse.org/papyrus/sysml/1.1/SysML",
    "http://www.eclipse.org/papyrus/sysml/1.1/SysML/Blocks",
    "http://www.eclipse.org/papyrus/sysml/1.1/SysML/Requirements",
    "http://www.omg.org/spec/SysML/20150709/SysML",
    "http://www.omg.org/spec/SysML/20181001/SysML",
    "http://www.omg.org/spec/SysML/",
    "http://www.nomagic.com/magicdraw/SysML/2.0",
})

# Known SysML namespace URI patterns for prefix matching
_SYSML_NS_PATTERNS = [
    re.compile(r"http://www\.eclipse\.org/papyrus/sysml"),
    re.compile(r"http://www\.omg\.org/spec/SysML"),
    re.compile(r"http://www\.nomagic\.com/.*[Ss]ys[Mm][Ll]"),
    re.compile(r"http://cameo\.nomagic\.com"),
    re.compile(r"http://www\.omg\.org/spec/UML"),
]

_XMI_NS = "http://www.omg.org/spec/XMI/20131001"
_XMI_NS_ALT = "http://www.omg.org/XMI"
_UML_NS_PATTERNS = [
    re.compile(r"http://www\.eclipse\.org/uml2"),
    re.compile(r"http://www\.omg\.org/spec/UML"),
    re.compile(r"http://www\.omg\.org/UML"),
]


def _strip_ns(tag: str) -> str:
    """Strip ``{http://...}`` namespace prefix from an XML tag name."""
    if tag and tag[0] == "{":
        return tag[tag.index("}") + 1:]
    return tag


def _is_sysml_ns(uri: str) -> bool:
    """Return True if *uri* looks like a SysML namespace URI."""
    if uri in _SYSML_PREFIXES:
        return True
    for pat in _SYSML_NS_PATTERNS:
        if pat.search(uri):
            return True
    return False


def _get_ns(tag: str) -> str:
    """Extract ``{http://...}`` namespace URI (without braces)."""
    if tag and tag[0] == "{":
        return tag[1:tag.index("}")]
    return ""


def _xmi_id(elem) -> str:
    """Return the xmi:id or xmi:ID attribute of an ElementTree element."""
    for ns in (_XMI_NS, _XMI_NS_ALT, ""):
        key = f"{{{ns}}}id" if ns else "xmi:id"
        val = elem.get(f"{{{ns}}}id" if ns else None) or elem.get(key)
        if val:
            return val
    # fallback: scan all attributes for something ending in ":id" or "xmi:id"
    for k, v in elem.attrib.items():
        if _strip_ns(k).lower() == "id" and v:
            return v
    return ""


def _xmi_type(elem) -> str:
    """Return the xmi:type attribute value (stripped of namespace prefix)."""
    for ns in (_XMI_NS, _XMI_NS_ALT):
        val = elem.get(f"{{{ns}}}type")
        if val:
            # "uml:Class" → "uml:Class"  (keep colon form for display)
            return val
    return elem.get("xmi:type", "")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class V1Element:
    """One model element from a SysML v1 XMI or Turtle model."""
    xmi_id: str
    name: str
    element_type: str        # e.g. "uml:Class", "uml:Activity"
    stereotype: str          # e.g. "Block", "Requirement", "" if none
    role: Role               # mapped IREB semantic role
    doc: str = ""            # from <ownedComment body="…"/> or rdfs:comment
    req_text: str = ""       # SysML:Requirement/@text or sysml:text
    req_id: str = ""         # SysML:Requirement/@id or sysml:id
    package: str = ""        # enclosing package name
    properties: dict = field(default_factory=dict)   # name → type_id


@dataclass
class V1Relation:
    """A directed relation between two V1Element nodes."""
    xmi_id: str
    source_id: str           # xmi:id of source element
    target_id: str           # xmi:id of target element
    rel_type: str            # "satisfy", "refine", "derive", "trace",
                             # "allocate", "composition", "association",
                             # "generalization", "flow"
    name: str = ""


@dataclass
class SysMLV1Model:
    """In-memory knowledge graph of a SysML v1 model."""
    elements: list[V1Element]
    relations: list[V1Relation]
    source_path: str = ""
    packages: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # KG export — GraphML
    # ------------------------------------------------------------------

    def to_graphml(self) -> str:
        """Serialize the full model as a GraphML knowledge graph."""
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<graphml xmlns="{ns}">',
            '  <key id="node_type"    for="node" attr.name="node_type"    attr.type="string"/>',
            '  <key id="name"         for="node" attr.name="name"         attr.type="string"/>',
            '  <key id="role"         for="node" attr.name="role"         attr.type="string"/>',
            '  <key id="stereotype"   for="node" attr.name="stereotype"   attr.type="string"/>',
            '  <key id="element_type" for="node" attr.name="element_type" attr.type="string"/>',
            '  <key id="doc"          for="node" attr.name="doc"          attr.type="string"/>',
            '  <key id="package"      for="node" attr.name="package"      attr.type="string"/>',
            '  <key id="req_id"       for="node" attr.name="req_id"       attr.type="string"/>',
            '  <key id="req_text"     for="node" attr.name="req_text"     attr.type="string"/>',
            '  <key id="xmi_id"       for="node" attr.name="xmi_id"       attr.type="string"/>',
            '  <key id="rel_type"     for="edge" attr.name="rel_type"     attr.type="string"/>',
            '  <key id="name"         for="edge" attr.name="name"         attr.type="string"/>',
            '  <graph id="sysml-v1-kg" edgedefault="directed">',
        ]

        from .core import _xml_escape

        for e in self.elements:
            ntype = "REQUIREMENT" if e.stereotype == "Requirement" else "MODEL_ELEMENT"
            eid = _xml_escape(e.xmi_id or e.name)
            lines.append(f'    <node id="{eid}">')
            lines.append(f'      <data key="node_type">{_xml_escape(ntype)}</data>')
            lines.append(f'      <data key="name">{_xml_escape(e.name)}</data>')
            lines.append(f'      <data key="role">{_xml_escape(e.role.value)}</data>')
            lines.append(f'      <data key="stereotype">{_xml_escape(e.stereotype)}</data>')
            lines.append(f'      <data key="element_type">{_xml_escape(e.element_type)}</data>')
            if e.doc:
                lines.append(f'      <data key="doc">{_xml_escape(e.doc)}</data>')
            if e.package:
                lines.append(f'      <data key="package">{_xml_escape(e.package)}</data>')
            if e.req_id:
                lines.append(f'      <data key="req_id">{_xml_escape(e.req_id)}</data>')
            if e.req_text:
                lines.append(f'      <data key="req_text">{_xml_escape(e.req_text)}</data>')
            lines.append(f'      <data key="xmi_id">{_xml_escape(e.xmi_id)}</data>')
            lines.append('    </node>')

        for i, r in enumerate(self.relations):
            src = _xml_escape(r.source_id)
            tgt = _xml_escape(r.target_id)
            lines.append(f'    <edge id="r{i}" source="{src}" target="{tgt}">')
            lines.append(f'      <data key="rel_type">{_xml_escape(r.rel_type)}</data>')
            if r.name:
                lines.append(f'      <data key="name">{_xml_escape(r.name)}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # KG export — canonical Turtle
    # ------------------------------------------------------------------

    def to_turtle(self) -> str:
        """Serialize the model as canonical Turtle/RDF using the sysmlkg ontology."""
        from .core import _ttl_literal

        KG = "sysmlkg"
        KG_NS = "http://reqgraph.io/sysml/"

        def _local(s: str) -> str:
            """Convert a name to a safe Turtle local part."""
            safe = re.sub(r"[^A-Za-z0-9_]", "_", s.strip()) or "elem"
            if safe[0].isdigit():
                safe = "e_" + safe
            return safe

        pkg_uris = {}
        for pkg in sorted(set(self.packages)):
            pkg_uris[pkg] = _local(pkg)

        # Stable subject ordering for deterministic output
        sorted_elems = sorted(self.elements, key=lambda e: e.xmi_id or e.name)
        sorted_rels = sorted(self.relations,
                             key=lambda r: (r.source_id, r.rel_type, r.target_id))

        # Map xmi_id → local name (prefer name, fall back to xmi_id)
        def _uri(elem: V1Element) -> str:
            n = _local(elem.name) if elem.name else _local(elem.xmi_id)
            return f"{KG}:{n}"

        id_to_elem = {e.xmi_id: e for e in self.elements if e.xmi_id}

        lines = [
            f"@prefix {KG}: <{KG_NS}> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
            "",
        ]

        for e in sorted_elems:
            st = e.stereotype or _strip_ns(e.element_type).replace(":", "_")
            uri = _uri(e)
            lines.append(f"{uri} a {KG}:{_local(st)} ;")
            lines.append(f'    rdfs:label {_ttl_literal(e.name)!r} ;')
            if e.doc:
                lines.append(f'    rdfs:comment {_ttl_literal(e.doc)!r} ;')
            if e.package:
                lines.append(f"    {KG}:inPackage {KG}:{_local(e.package)} ;")
            if e.req_id:
                lines.append(f'    {KG}:requirementId {_ttl_literal(e.req_id)!r} ;')
            if e.req_text:
                lines.append(f'    {KG}:requirementText {_ttl_literal(e.req_text)!r} ;')
            # Trim trailing " ;" and replace with " ."
            lines[-1] = lines[-1][:-2] + " ."
            lines.append("")

        for r in sorted_rels:
            src_elem = id_to_elem.get(r.source_id)
            tgt_elem = id_to_elem.get(r.target_id)
            if not src_elem or not tgt_elem:
                continue
            src_uri = _uri(src_elem)
            tgt_uri = _uri(tgt_elem)
            pred = _local(r.rel_type)
            lines.append(f"{src_uri} {KG}:{pred} {tgt_uri} .")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------

def _map_role(uml_type: str, stereotype: str) -> Role:
    """Map (UML type, SysML stereotype) to an IREB semantic Role."""
    st = stereotype.lower()
    ut = uml_type.lower()

    # Stereotype-first overrides
    if st in ("block", "constraintblock", "sysmlblock"):
        return Role.SUBJECT
    if st in ("requirement", "requirementusage", "requirementdefinition"):
        return Role.CONSTRAINT
    if st in ("flowport", "proxyport", "fullport", "portusage", "portdefinition"):
        return Role.ACTOR
    if st in ("flowproperty", "valueproperty", "attributeusage", "datatype"):
        return Role.OBJECT
    if st in ("satisfy", "refine", "derive", "trace", "allocate"):
        return Role.CONSTRAINT  # will be turned into a V1Relation, not element

    # UML-type fallback
    if "activity" in ut or "action" in ut or "behavior" in ut:
        return Role.PROCESS
    if "statemachine" in ut or "state" in ut or "pseudostate" in ut or "region" in ut:
        return Role.CONDITION
    if "port" in ut:
        return Role.ACTOR
    if "property" in ut or "attribute" in ut:
        return Role.OBJECT
    if "constraint" in ut:
        return Role.CONSTRAINT
    if "class" in ut or "component" in ut or "node" in ut or "artifact" in ut:
        return Role.SUBJECT
    if "interface" in ut:
        return Role.ACTOR
    if "signal" in ut or "operation" in ut or "reception" in ut:
        return Role.PROCESS

    return Role.SUBJECT  # safe default


# ---------------------------------------------------------------------------
# XMI parser (two-pass)
# ---------------------------------------------------------------------------

_ABSTRACTION_TYPES = frozenset({
    "uml:Abstraction", "Abstraction",
    "uml:Realization", "Realization",
    "uml:Usage", "Usage",
})

_SKIP_UML_TYPES = frozenset({
    "uml:Package", "Package",
    "uml:Model", "Model",
    "uml:Profile", "Profile",
    "uml:Stereotype", "Stereotype",
    "uml:Extension", "Extension",
    "uml:ExtensionEnd",
})

_RELATION_STEREOTYPE_MAP = {
    "satisfy": "satisfy",
    "refine": "refine",
    "derive": "derive",
    "derivereqt": "derive",
    "trace": "trace",
    "allocate": "allocate",
}


def _parse_xmi(text: str, source_path: str = "") -> SysMLV1Model:
    """Two-pass XMI parser for SysML v1 models."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise DataFormatError(f"XMI parse error: {exc}") from exc

    # --- Pass 1: collect all packagedElements by xmi:id -------------------

    # raw_elems[xmi_id] = {tag, attrib, doc, properties, relations}
    raw: dict[str, dict] = {}
    # abstractions[xmi_id] = {client: xmi_id, supplier: xmi_id}
    abstractions: dict[str, dict] = {}
    # generalizations[xmi_id] = {specific: xmi_id, general: xmi_id}
    generalizations: dict[str, dict] = {}
    # package membership xmi_id -> package_name
    pkg_map: dict[str, str] = {}
    packages: list[str] = []

    def _collect(elem, current_pkg=""):
        """Recursively walk the element tree collecting elements and relations."""
        local = _strip_ns(elem.tag)
        eid = _xmi_id(elem)
        etype = _xmi_type(elem) or elem.tag  # e.g. "uml:Class"
        etype_local = _strip_ns(etype.split(":")[-1] if ":" in etype else etype)

        # Package tracking
        if etype in ("uml:Package", "uml:Model") or etype_local in ("Package", "Model"):
            pkg_name = elem.get("name", "")
            if pkg_name and pkg_name not in packages:
                packages.append(pkg_name)
            if eid:
                raw[eid] = {"name": pkg_name, "type": etype, "is_pkg": True,
                            "doc": "", "properties": {}, "current_pkg": pkg_name}
            for child in elem:
                _collect(child, current_pkg=pkg_name or current_pkg)
            return

        if local in ("packagedElement", "ownedMember", "ownedBehavior",
                     "nestedClassifier", "ownedElement", "member"):
            # Abstraction → collect client/supplier
            if etype in _ABSTRACTION_TYPES or etype_local == "Abstraction":
                client_ids = [c.get("{%s}idref" % _XMI_NS) or c.get("xmi:idref", "")
                              for c in elem if _strip_ns(c.tag) == "client"]
                supplier_ids = [c.get("{%s}idref" % _XMI_NS) or c.get("xmi:idref", "")
                                for c in elem if _strip_ns(c.tag) == "supplier"]
                if not eid:
                    eid = elem.get("name", f"_abs_{len(abstractions)}")
                abstractions[eid] = {
                    "clients": [i for i in client_ids if i],
                    "suppliers": [i for i in supplier_ids if i],
                    "xmi_id": eid,
                }

            # Generalization
            elif etype in ("uml:Generalization", "Generalization") or etype_local == "Generalization":
                specific = elem.get("specific") or current_pkg
                general = elem.get("general", "")
                if eid:
                    generalizations[eid] = {"specific": specific, "general": general}

            elif etype in _SKIP_UML_TYPES or etype_local in ("Package", "Model", "Profile"):
                pkg_name = elem.get("name", "")
                if pkg_name and pkg_name not in packages:
                    packages.append(pkg_name)
                for child in elem:
                    _collect(child, current_pkg=pkg_name or current_pkg)
                return

            else:
                if eid:
                    name = elem.get("name", "")
                    doc = ""
                    props = {}
                    for child in elem:
                        child_local = _strip_ns(child.tag)
                        if child_local == "ownedComment":
                            doc = child.get("body", "") or child.get(
                                "{http://www.omg.org/spec/XMI/20131001}body", "")
                        elif child_local == "ownedAttribute":
                            pname = child.get("name", "")
                            ptype = (child.get("type") or
                                     child.get("{%s}idref" % _XMI_NS, ""))
                            if pname:
                                props[pname] = ptype
                        # nested elements (sub-activities, states, etc.)
                        _collect(child, current_pkg=current_pkg)

                    raw[eid] = {"name": name, "type": etype, "is_pkg": False,
                                "doc": doc, "properties": props,
                                "current_pkg": current_pkg}
                    pkg_map[eid] = current_pkg

        elif local in ("ownedAttribute", "ownedParameter"):
            if eid:
                name = elem.get("name", "")
                raw[eid] = {"name": name, "type": etype or "uml:Property",
                            "is_pkg": False, "doc": "",
                            "properties": {}, "current_pkg": current_pkg}
                pkg_map[eid] = current_pkg

        elif local in ("subvertex", "region", "ownedTransition"):
            if eid:
                name = elem.get("name", "")
                raw[eid] = {"name": name, "type": etype or "uml:State",
                            "is_pkg": False, "doc": "",
                            "properties": {}, "current_pkg": current_pkg}
                pkg_map[eid] = current_pkg

        # Recurse into all children
        for child in elem:
            _collect(child, current_pkg=current_pkg)

    _collect(root)

    # --- Pass 2: apply stereotypes ----------------------------------------

    # stereotype_map[base_id] = {stereotype_name, req_text, req_id, ...}
    stereotype_map: dict[str, dict] = {}
    # abstraction_stereotypes[abs_xmi_id] = stereotype_name
    abstraction_stereo: dict[str, str] = {}

    for child in root.iter():
        tag_ns = _get_ns(child.tag)
        if not tag_ns or not _is_sysml_ns(tag_ns):
            continue
        stereo_name = _strip_ns(child.tag)
        if not stereo_name or stereo_name.lower() in ("model", "package"):
            continue

        # Find which element this stereotype applies to
        base_id = None
        for attr_name in ("base_Class", "base_Activity", "base_Property",
                          "base_Port", "base_Constraint", "base_StateMachine",
                          "base_State", "base_Element", "base_NamedElement"):
            val = child.get(attr_name)
            if val:
                base_id = val
                break

        base_abs_id = child.get("base_Abstraction")

        if base_id:
            if base_id not in stereotype_map:
                stereotype_map[base_id] = {}
            stereotype_map[base_id]["stereotype"] = stereo_name
            # Requirement-specific attributes
            if stereo_name.lower() in ("requirement",):
                for attr in ("text", "Text"):
                    v = child.get(attr)
                    if v:
                        stereotype_map[base_id]["req_text"] = v
                        break
                for attr in ("id", "Id"):
                    v = child.get(attr)
                    if v:
                        stereotype_map[base_id]["req_id"] = v
                        break

        if base_abs_id:
            abstraction_stereo[base_abs_id] = stereo_name

    # --- Build elements list ----------------------------------------------

    elements: list[V1Element] = []
    for eid, info in raw.items():
        if info.get("is_pkg"):
            continue
        name = info.get("name", "")
        etype = info.get("type", "")

        stereo_info = stereotype_map.get(eid, {})
        stereotype = stereo_info.get("stereotype", "")

        role = _map_role(etype, stereotype)

        elem = V1Element(
            xmi_id=eid,
            name=name,
            element_type=etype,
            stereotype=stereotype,
            role=role,
            doc=info.get("doc", ""),
            req_text=stereo_info.get("req_text", ""),
            req_id=stereo_info.get("req_id", ""),
            package=info.get("current_pkg", ""),
            properties=info.get("properties", {}),
        )
        elements.append(elem)

    # --- Build relations list ---------------------------------------------

    relations: list[V1Relation] = []

    # Abstraction-based relations (satisfy, refine, etc.)
    for abs_id, abs_info in abstractions.items():
        stereo = abstraction_stereo.get(abs_id, "").lower()
        rel_type = _RELATION_STEREOTYPE_MAP.get(stereo, "association")

        for client_id in abs_info["clients"]:
            for supplier_id in abs_info["suppliers"]:
                if client_id and supplier_id:
                    relations.append(V1Relation(
                        xmi_id=abs_id,
                        source_id=client_id,
                        target_id=supplier_id,
                        rel_type=rel_type,
                    ))

    # Generalizations
    for gen_id, gen_info in generalizations.items():
        spec = gen_info.get("specific", "")
        gen = gen_info.get("general", "")
        if spec and gen:
            relations.append(V1Relation(
                xmi_id=gen_id,
                source_id=spec,
                target_id=gen,
                rel_type="generalization",
            ))

    # Composition from ownedAttribute aggregation
    for eid, info in raw.items():
        if info.get("is_pkg"):
            continue
        for prop_name, prop_type_id in info.get("properties", {}).items():
            if prop_type_id and prop_type_id in raw:
                relations.append(V1Relation(
                    xmi_id=f"{eid}_{prop_name}",
                    source_id=eid,
                    target_id=prop_type_id,
                    rel_type="composition",
                    name=prop_name,
                ))

    return SysMLV1Model(
        elements=elements,
        relations=relations,
        source_path=source_path,
        packages=packages,
    )


# ---------------------------------------------------------------------------
# Turtle / RDF parser
# ---------------------------------------------------------------------------

# Maps RDF type local names to IREB roles
_RDF_TYPE_ROLE: dict[str, Role] = {
    "block": Role.SUBJECT,
    "constraintblock": Role.SUBJECT,
    "sysmlblock": Role.SUBJECT,
    "component": Role.SUBJECT,
    "subsystem": Role.SUBJECT,
    "systemblock": Role.SUBJECT,
    "actionusage": Role.PROCESS,
    "actiondefinition": Role.PROCESS,
    "activity": Role.PROCESS,
    "behaviorusage": Role.PROCESS,
    "behaviordefinition": Role.PROCESS,
    "transition": Role.PROCESS,
    "flowproperty": Role.OBJECT,
    "attributeusage": Role.OBJECT,
    "valueproperty": Role.OBJECT,
    "dataproperty": Role.OBJECT,
    "attribute": Role.OBJECT,
    "statedefinition": Role.CONDITION,
    "stateusage": Role.CONDITION,
    "statemachinedefinition": Role.CONDITION,
    "state": Role.CONDITION,
    "flowport": Role.ACTOR,
    "portusage": Role.ACTOR,
    "portdefinition": Role.ACTOR,
    "proxyport": Role.ACTOR,
    "fullport": Role.ACTOR,
    "port": Role.ACTOR,
    "requirement": Role.CONSTRAINT,
    "requirementusage": Role.CONSTRAINT,
    "requirementdefinition": Role.CONSTRAINT,
    "constraint": Role.CONSTRAINT,
}

# Predicate local names that map to V1Relation types
_RDF_PRED_REL: dict[str, str] = {
    "satisfies": "satisfy",
    "satisfy": "satisfy",
    "refines": "refine",
    "refine": "refine",
    "derives": "derive",
    "derive": "derive",
    "traces": "trace",
    "trace": "trace",
    "allocates": "allocate",
    "allocate": "allocate",
    "hasproperty": "composition",
    "haspart": "composition",
    "hasport": "association",
    "ownedattribute": "composition",
    "containspart": "composition",
}


def _parse_turtle(text: str, source_path: str = "") -> SysMLV1Model:
    """Parse a Turtle/RDF SysML ontology into SysMLV1Model.

    Requires ``rdflib>=6.0``.
    """
    try:
        import rdflib
        from rdflib import RDF, RDFS, OWL, Literal
        from rdflib.namespace import Namespace
    except ImportError as exc:
        raise ImportError(
            "rdflib is required for Turtle/RDF input. "
            "Install it with: pip install rdflib>=6.0"
        ) from exc

    g = rdflib.Graph()
    try:
        g.parse(data=text, format="turtle")
    except Exception as exc:
        raise DataFormatError(f"Turtle/RDF parse error: {exc}") from exc

    RDF_TYPE = RDF.type
    RDFS_LABEL = RDFS.label
    RDFS_COMMENT = RDFS.comment
    RDFS_SUBCLASSOF = RDFS.subClassOf

    # --- Step 1: collect all rdf:type(s) per subject, resolve role --------

    # subject_types[uri] = [local_name, ...]
    subject_types: dict = {}
    subject_labels: dict = {}
    subject_comments: dict = {}
    subject_req_text: dict = {}
    subject_req_id: dict = {}
    subject_package: dict = {}

    def _local_name(uri) -> str:
        s = str(uri)
        if "#" in s:
            return s.split("#")[-1]
        return s.rstrip("/").split("/")[-1]

    for s, p, o in g:
        su = str(s)
        if p == RDF_TYPE:
            subject_types.setdefault(su, []).append(_local_name(o).lower())
        elif p == RDFS_LABEL:
            subject_labels[su] = str(o)
        elif p == RDFS_COMMENT:
            subject_comments[su] = str(o)
        else:
            plocal = _local_name(p).lower()
            if plocal in ("text", "requirementtext", "req_text"):
                subject_req_text[su] = str(o)
            elif plocal in ("id", "requirementid", "req_id"):
                subject_req_id[su] = str(o)
            elif plocal in ("inpackage", "package", "namespace"):
                subject_package[su] = _local_name(o)

    # Build inheritance map for custom stereotypes
    # subclass_map[child_local] = parent_local
    subclass_map: dict[str, str] = {}
    for s, p, o in g:
        if p == RDFS_SUBCLASSOF:
            child = _local_name(s).lower()
            parent = _local_name(o).lower()
            subclass_map[child] = parent

    def _resolve_role(type_locals: list[str]) -> tuple[str, Role]:
        """Return (stereotype, role) for a list of rdf:type local names."""
        for tl in type_locals:
            if tl in _RDF_TYPE_ROLE:
                return tl.capitalize(), _RDF_TYPE_ROLE[tl]
        # Try subclass chain
        for tl in type_locals:
            parent = subclass_map.get(tl)
            while parent:
                if parent in _RDF_TYPE_ROLE:
                    return tl.capitalize(), _RDF_TYPE_ROLE[parent]
                parent = subclass_map.get(parent)
        # Filter out OWL/RDF meta-types
        useful = [t for t in type_locals
                  if t not in ("class", "thing", "namedindividual", "ontology",
                               "objectproperty", "datatypeproperty", "annotation")]
        stereotype = useful[0].capitalize() if useful else ""
        return stereotype, Role.SUBJECT

    # --- Step 2: build elements -------------------------------------------

    elements: list[V1Element] = []
    packages = set()

    for su, type_locals in subject_types.items():
        # Skip pure meta-subjects (rdf:type, owl:Ontology, etc.)
        if not type_locals or all(t in ("class", "thing", "namedindividual",
                                        "ontology", "objectproperty",
                                        "datatypeproperty", "annotation")
                                  for t in type_locals):
            continue

        local_id = _local_name(su)
        name = subject_labels.get(su, local_id)
        stereotype, role = _resolve_role(type_locals)
        pkg = subject_package.get(su, "")
        if pkg:
            packages.add(pkg)

        elem = V1Element(
            xmi_id=local_id,
            name=name,
            element_type=type_locals[0] if type_locals else "",
            stereotype=stereotype,
            role=role,
            doc=subject_comments.get(su, ""),
            req_text=subject_req_text.get(su, ""),
            req_id=subject_req_id.get(su, ""),
            package=pkg,
        )
        elements.append(elem)

    # --- Step 3: build relations ------------------------------------------

    relations: list[V1Relation] = []
    elem_ids = {e.xmi_id for e in elements}

    for s, p, o in g:
        plocal = _local_name(p).lower()
        rel_type = _RDF_PRED_REL.get(plocal)
        if not rel_type:
            continue
        src_local = _local_name(s)
        tgt_local = _local_name(o)
        if src_local not in elem_ids or tgt_local not in elem_ids:
            continue
        rel_id = f"{src_local}_{plocal}_{tgt_local}"
        relations.append(V1Relation(
            xmi_id=rel_id,
            source_id=src_local,
            target_id=tgt_local,
            rel_type=rel_type,
        ))

    return SysMLV1Model(
        elements=elements,
        relations=relations,
        source_path=source_path,
        packages=sorted(packages),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_format(text: str, path: str = "") -> str:
    """Auto-detect 'turtle' or 'xmi' from file extension + content sniffing."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("ttl", "rdf", "n3", "owl"):
        return "turtle"
    if ext in ("xmi", "uml", "xml"):
        return "xmi"
    # Content sniffing
    head = text[:200].lstrip()
    if head.startswith("@prefix") or head.startswith("@base") or "@prefix" in head[:100]:
        return "turtle"
    if head.startswith("<?xml") or head.startswith("<uml:") or "<uml:" in head:
        return "xmi"
    return "xmi"  # safe default


def parse_sysml_v1(text: str, source_path: str = "") -> SysMLV1Model:
    """Parse SysML v1 XMI or Turtle text into a ``SysMLV1Model``.

    Format is auto-detected from the content and *source_path* extension.
    """
    if not text or not text.strip():
        raise DataFormatError("empty SysML v1 input")
    fmt = _detect_format(text, source_path)
    if fmt == "turtle":
        return _parse_turtle(text, source_path)
    return _parse_xmi(text, source_path)


def read_sysml_v1(path: str) -> SysMLV1Model:
    """Read a SysML v1 XMI or Turtle file from *path*.

    Auto-detects format by extension and content sniffing.
    Raises ``DataFormatError`` for empty / unreadable files.
    """
    import os
    if not os.path.isfile(path):
        raise DataFormatError(f"SysML v1 model file not found: {path!r}")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise DataFormatError(f"cannot read {path!r}: {exc}") from exc
    if not text.strip():
        raise DataFormatError(f"SysML v1 file is empty: {path!r}")
    return parse_sysml_v1(text, source_path=path)
