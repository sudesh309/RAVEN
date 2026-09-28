"""
reqgraph.io_formats
===================

Batch import / export of requirement sets:

* CSV / Excel  (pandas + openpyxl) -- a flat table with one decomposed column
  per requirement element, plus type / EARS / quality columns.
* ReqIF        (lxml) -- the OMG Requirements Interchange Format used by DOORS,
  Polarion, etc. A minimal, well-formed subset (ID + text) is written/read;
  export->import round-trips the requirement text.
* JSON         (stdlib json + pandas) -- array-of-objects or dict-keyed shapes.

All readers return 3-tuples ``(id, text, metadata_dict)`` where ``metadata_dict``
carries any extra columns beyond ``id`` and ``text`` (e.g. rationale,
applicability, additional_info). Column names are normalised to lowercase with
spaces/hyphens replaced by underscores. Callers that only need (id, text) can
unpack as ``for rid, text, *_ in items``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Iterable, Optional

from .core import Role
from .errors import DataFormatError
from .parser import RequirementParser
from .templates import RUPP_TEMPLATE, Template

logger = logging.getLogger(__name__)

_ELEMENT_COLUMNS = [Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
                    Role.PROCESS, Role.OBJECT, Role.CONSTRAINT]

# Output columns produced by the parser/quality analysis. A metadata column
# whose normalised name collides with one of these would otherwise be clobbered,
# so it is namespaced with an ``attr_`` prefix (see _resolve_meta_columns).
_RESERVED_COLUMNS = ({"id", "text", "type", "ears_pattern", "roundtrip_ok",
                      "error", "weak_words", "non_atomic"}
                     | {r.value.lower() for r in _ELEMENT_COLUMNS})

_COL_NORM_RE = re.compile(r"[\s\-]+")


def _col_norm(name: str) -> str:
    """Normalise a column name: lowercase, spaces/hyphens → underscore."""
    return _COL_NORM_RE.sub("_", name.strip().lower())


def _resolve_meta_columns(meta_keys) -> dict:
    """Map metadata keys → output column names, namespacing reserved collisions.

    A source attribute called e.g. "Subject" or "Type" would collide with the
    parsed SUBJECT element / computed requirement type column; such keys are
    emitted as ``attr_subject`` / ``attr_type`` so no user data is lost. Order
    is preserved.
    """
    out = {}
    for k in dict.fromkeys(meta_keys):
        out[k] = f"attr_{k}" if k in _RESERVED_COLUMNS else k
    return out


# --- predefined-format column conventions -----------------------------------
#
# Requirement documents rarely use the literal headers "id" and "text": DOORS
# exports "Object Text"/"Object Identifier", Polarion and Jama use "Description"
# / "Requirement ID", hand-written tables say "Requirement". Recognising the
# conventions real tools emit is what lets an unmodified export load as-is.
# Names are compared after _col_norm (lowercase, spaces/hyphens → underscore).
_TEXT_ALIASES = (
    "text", "requirement", "requirement_text", "requirementtext", "req_text",
    "reqtext", "reqif.text", "object_text", "description", "statement",
    "requirement_description", "requirement_statement", "body", "content",
    "shall_statement", "primary_text", "specification",
)
_ID_ALIASES = (
    "id", "req_id", "reqid", "requirement_id", "requirementid", "identifier",
    "reqif.foreignid", "object_identifier", "object_id", "absolute_number",
    "key", "tag", "reference", "req_no", "requirement_no", "number", "article",
)


def _pick_column(columns, explicit, aliases, kind):
    """Resolve the column holding the id / text, tolerating naming conventions.

    ``explicit`` wins when present (exact name, then normalised). Otherwise the
    first alias that matches a column is used, preferring earlier aliases so
    "text" beats "description" when a file happens to carry both.
    """
    cols = list(columns)
    norm = {}
    for c in cols:
        norm.setdefault(_col_norm(str(c)), c)

    if explicit is not None:
        if explicit in cols:
            return explicit
        hit = norm.get(_col_norm(str(explicit)))
        if hit is not None:
            return hit
        # an explicitly requested column that is genuinely absent is an error
        # for text (nothing to parse) but merely "no ids" for the id column
        if kind == "text" and explicit not in _TEXT_ALIASES[:1]:
            raise DataFormatError(
                f"text column {explicit!r} not found; available columns: {cols}")

    for alias in aliases:
        hit = norm.get(alias)
        if hit is not None:
            return hit
    return None


def _normalise(items):
    """Accept str, (id, text) or (id, text, meta); yield (id, text)."""
    for it in items:
        if isinstance(it, (tuple, list)):
            yield (it[0], it[1])
        else:
            yield (None, it)


# ---------------------------------------------------------------------------
# CSV / Excel
# ---------------------------------------------------------------------------

def requirements_to_dataframe(items: Iterable, template: Template = RUPP_TEMPLATE,
                              extractor=None, analyze: bool = True):
    """Decompose a requirement set into a flat table.

    Accepts str, (id, text) or (id, text, meta_dict) items. Extra metadata
    fields (rationale, applicability, etc.) are injected as columns between
    ``text`` and ``type`` in the output DataFrame.

    Fault isolation: a row that fails to parse is recorded with its error
    message instead of aborting the whole batch."""
    import pandas as pd
    from .quality import enrich

    # materialise once so we can scan meta keys and iterate again
    item_list = list(items)

    # discover all extra metadata keys in insertion order, resolving collisions
    # with reserved output columns (e.g. a "Subject" attribute → attr_subject)
    meta_cols = _resolve_meta_columns(
        k
        for it in item_list
        if isinstance(it, (tuple, list)) and len(it) >= 3 and isinstance(it[2], dict)
        for k in it[2]
    )

    parser = RequirementParser(template, extractor)
    rows = []
    for it in item_list:
        if isinstance(it, (tuple, list)):
            rid, text = it[0], it[1]
            meta = it[2] if len(it) >= 3 and isinstance(it[2], dict) else {}
        else:
            rid, text, meta = None, it, {}

        row = {"id": rid, "text": text}
        # inject extra metadata columns before quality columns
        for k, col in meta_cols.items():
            row[col] = meta.get(k, "")
        row.update({"type": None, "ears_pattern": None,
                    "roundtrip_ok": False, "error": ""})
        for r in _ELEMENT_COLUMNS:
            row[r.value.lower()] = ""
        row["weak_words"] = ""
        row["non_atomic"] = ""
        try:
            g = parser.parse(text, metadata={"id": rid} if rid else None)
            if analyze:
                enrich(g)
            bucket = {}
            for n in g.elements():
                bucket.setdefault(n.role.value, []).append(n.text.strip())
            row.update({"type": g.analysis.get("type"),
                        "ears_pattern": g.analysis.get("ears_pattern"),
                        "roundtrip_ok": g.generate() == text})
            for r in _ELEMENT_COLUMNS:
                row[r.value.lower()] = " | ".join(bucket.get(r.value, []))
            q = g.analysis.get("quality", {})
            row["weak_words"] = ", ".join(q.get("weak_words", []))
            row["non_atomic"] = q.get("non_atomic", "")
        except Exception as exc:  # keep the batch alive; surface the failure
            logger.error("failed to process requirement %r: %s", rid or text[:40], exc)
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(items, path, **kw):
    requirements_to_dataframe(items, **kw).to_csv(path, index=False)
    return path


def write_excel(items, path, **kw):
    requirements_to_dataframe(items, **kw).to_excel(path, index=False)
    return path


def read_requirements_csv(path, text_column="text", id_column="id"):
    import pandas as pd
    df = pd.read_csv(path)
    return _rows_from_df(df, text_column, id_column)


def read_requirements_excel(path, text_column="text", id_column="id"):
    import pandas as pd
    df = pd.read_excel(path)
    return _rows_from_df(df, text_column, id_column)


def read_requirements_json(path, text_column="text", id_column="id"):
    """Read requirements from a JSON file.

    Supports two shapes:
      - Array of objects:  [{"id": "R1", "text": "...", "rationale": "..."}, ...]
      - Dict-keyed:        {"R1": {"text": "...", ...}, ...}
                        or {"R1": "plain text", ...}

    Returns a list of 3-tuples (id, text, metadata_dict).
    """
    import json as _json
    import pandas as pd

    with open(path, encoding="utf-8") as fh:
        raw = _json.load(fh)

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict):
        rows = []
        for key, val in raw.items():
            if isinstance(val, str):
                rows.append({id_column: key, text_column: val})
            elif isinstance(val, dict):
                row = dict(val)
                if id_column not in row:
                    row[id_column] = key
                rows.append(row)
            else:
                raise DataFormatError(
                    f"JSON dict values must be str or object, got "
                    f"{type(val).__name__!r} for key {key!r}")
        df = pd.DataFrame(rows)
    else:
        raise DataFormatError(
            f"JSON root must be an array or object, got {type(raw).__name__!r}")

    return _rows_from_df(df, text_column, id_column)


def _rows_from_df(df, text_column, id_column):
    """Extract rows as 3-tuples (id, text, metadata_dict).

    ``text_column``/``id_column`` are treated as *preferences*: if the file uses
    one of the conventional names instead (Object Text, Description, Req ID…),
    that column is used. Extra columns beyond id/text are normalised (lowercase,
    spaces→underscore) and returned in the metadata dict; NaN values are omitted.
    """
    import pandas as pd
    text_col = _pick_column(df.columns, text_column, _TEXT_ALIASES, "text")
    if text_col is None:
        raise DataFormatError(
            f"no requirement text column found; looked for {text_column!r} and "
            f"common names ({', '.join(_TEXT_ALIASES[:6])}…). "
            f"Available columns: {list(df.columns)}")
    id_col = _pick_column(df.columns, id_column, _ID_ALIASES, "id")

    extra_cols = [c for c in df.columns if c not in (text_col, id_col)]
    norm_map = {c: _col_norm(str(c)) for c in extra_cols}   # original → normalised

    items = []
    for _, r in df.iterrows():
        txt = r[text_col]
        if pd.isna(txt) or not str(txt).strip():
            continue                      # skip blank rows quietly
        rid = r[id_col] if id_col is not None else None
        if rid is not None and pd.isna(rid):
            rid = None
        meta = {}
        for orig, norm in norm_map.items():
            v = r[orig]
            if not pd.isna(v):
                meta[norm] = str(v)
        items.append((None if rid is None else str(rid), str(txt), meta))
    return items


# ---------------------------------------------------------------------------
# Word (.docx)
# ---------------------------------------------------------------------------

_DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag):
    return f"{{{_DOCX_NS}}}{tag}"


def _docx_para_text(p) -> str:
    """Visible text of one <w:p>, joining runs and honouring tabs/breaks."""
    import xml.etree.ElementTree as ET
    parts = []
    for node in p.iter():
        if node.tag == _w("t"):
            parts.append(node.text or "")
        elif node.tag == _w("tab"):
            parts.append("\t")
        elif node.tag in (_w("br"), _w("cr")):
            parts.append(" ")
    return "".join(parts).strip()


# "REQ-001:", "[SYS-12]", "R1 -", "3.1.2" ... an ID leading a requirement line
_DOCX_ID_RE = re.compile(
    r"^\s*(?:\[|\()?\s*"
    r"(?P<id>(?:[A-Z][A-Za-z]{0,9}[-_\.]\d+(?:[-_\.]\d+)*)"      # REQ-001, SYS_1.2
    r"|(?:[A-Z]{1,6}\d{1,6})"                                     # R1, SYS12
    r"|(?:\d+(?:\.\d+){1,5}))"                                    # 3.1.2
    r"\s*(?:\]|\))?\s*(?:[:\.\)\-–—]|\s)\s*(?P<rest>\S.*)$")

_MODALITY_LINE_RE = re.compile(
    r"\b(shall|must|should|will\s+be\s+able\s+to|is\s+required\s+to)\b", re.I)


def read_requirements_docx(path, text_column="text", id_column="id"):
    """Read requirements from a Word ``.docx`` file.

    Two predefined layouts are recognised, in this order:

    1. **Requirements table** -- any table whose header row names a requirement
       text column (``Requirement``, ``Object Text``, ``Description``…). Extra
       columns become metadata, exactly like a CSV. This is the layout most
       requirement documents and templates use.
    2. **ID-prefixed paragraphs** -- lines such as ``REQ-001: The system shall …``
       or ``3.1.2 The system shall …``. Used when no qualifying table is found.
       If a document has neither, every paragraph containing a modality
       (shall/must/should) is taken as one requirement, so prose specifications
       still import rather than failing.

    Reads the OOXML package directly with the stdlib (zipfile + ElementTree), so
    it needs no extra dependency. Returns ``(id, text, metadata_dict)`` triples.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "word/document.xml" not in names:
                raise DataFormatError(
                    "not a Word document: the archive has no word/document.xml "
                    "(a .doc file must be saved as .docx first)")
            xml = z.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise DataFormatError(
            "not a readable .docx file (it is not a valid OOXML package); "
            "legacy .doc must be re-saved as .docx") from exc

    body = ET.fromstring(xml).find(_w("body"))
    if body is None:
        raise DataFormatError("Word document has no body")

    # --- 1. a table whose header names a requirement text column -------------
    for tbl in body.iter(_w("tbl")):
        rows = []
        for tr in tbl.findall(_w("tr")):
            cells = [" ".join(_docx_para_text(p) for p in tc.iter(_w("p"))).strip()
                     for tc in tr.findall(_w("tc"))]
            if any(c for c in cells):
                rows.append(cells)
        if len(rows) < 2:
            continue
        header = rows[0]
        text_col = _pick_column(header, None, _TEXT_ALIASES, "text")
        if text_col is None:
            continue                      # not a requirements table; keep looking
        id_col = _pick_column(header, None, _ID_ALIASES, "id")
        ti, ii = header.index(text_col), (header.index(id_col) if id_col else None)
        meta_idx = {i: _col_norm(h) for i, h in enumerate(header)
                    if i not in (ti, ii) and h}

        items = []
        for cells in rows[1:]:
            if ti >= len(cells) or not cells[ti].strip():
                continue
            rid = (cells[ii].strip() or None) if ii is not None and ii < len(cells) else None
            meta = {name: cells[i].strip() for i, name in meta_idx.items()
                    if i < len(cells) and cells[i].strip()}
            items.append((rid, cells[ti].strip(), meta))
        if items:
            return items

    # --- 2. ID-prefixed paragraphs -------------------------------------------
    paras = [t for t in (_docx_para_text(p) for p in body.iter(_w("p"))) if t]
    items, prose = [], []
    for line in paras:
        m = _DOCX_ID_RE.match(line)
        if m and _MODALITY_LINE_RE.search(m.group("rest")):
            items.append((m.group("id"), m.group("rest").strip(), {}))
        elif _MODALITY_LINE_RE.search(line):
            prose.append(line)
    if items:
        return items
    # --- 3. fall back to any paragraph that reads like a requirement ---------
    if prose:
        return [(None, line, {}) for line in prose]

    raise DataFormatError(
        "no requirements found in this Word document: expected a table with a "
        "requirement text column (Requirement / Object Text / Description…), "
        "paragraphs prefixed with an ID (e.g. 'REQ-001: The system shall …'), "
        "or at least one paragraph containing shall/must/should")


# ---------------------------------------------------------------------------
# ReqIF (minimal, well-formed subset)
# ---------------------------------------------------------------------------

_REQIF_NS = "http://www.omg.org/spec/ReqIF/20110401/reqif.xsd"
_XHTML_NS = "http://www.w3.org/1999/xhtml"


def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _q(tag):
    return f"{{{_REQIF_NS}}}{tag}"


def write_reqif(items, path, title="reqgraph export"):
    """Export requirements to a minimal ReqIF XML file."""
    from lxml import etree

    root = etree.Element(_q("REQ-IF"), nsmap={None: _REQIF_NS})
    header = etree.SubElement(root, _q("THE-HEADER"))
    h = etree.SubElement(header, _q("REQ-IF-HEADER"), IDENTIFIER="hdr-1")
    etree.SubElement(h, _q("CREATION-TIME")).text = _now()
    etree.SubElement(h, _q("REQ-IF-TOOL-ID")).text = "reqgraph"
    etree.SubElement(h, _q("REQ-IF-VERSION")).text = "1.0"
    etree.SubElement(h, _q("SOURCE-TOOL-ID")).text = "reqgraph"
    etree.SubElement(h, _q("TITLE")).text = title

    content = etree.SubElement(root, _q("CORE-CONTENT"))
    rc = etree.SubElement(content, _q("REQ-IF-CONTENT"))

    # datatype (ReqIF attribute names contain hyphens, so set() them explicitly)
    dts = etree.SubElement(rc, _q("DATATYPES"))
    dt = etree.SubElement(dts, _q("DATATYPE-DEFINITION-STRING"))
    dt.set("IDENTIFIER", "DT-STRING")
    dt.set("LONG-NAME", "String")
    dt.set("MAX-LENGTH", "10000")
    dt.set("LAST-CHANGE", _now())

    # spec-object type with two string attributes (ID, Text)
    sts = etree.SubElement(rc, _q("SPEC-TYPES"))
    sot = etree.SubElement(sts, _q("SPEC-OBJECT-TYPE"))
    sot.set("IDENTIFIER", "ST-REQ")
    sot.set("LONG-NAME", "Requirement")
    sot.set("LAST-CHANGE", _now())
    sas = etree.SubElement(sot, _q("SPEC-ATTRIBUTES"))
    for ident, name in (("AD-ID", "ReqID"), ("AD-TEXT", "ReqText")):
        ad = etree.SubElement(sas, _q("ATTRIBUTE-DEFINITION-STRING"))
        ad.set("IDENTIFIER", ident)
        ad.set("LONG-NAME", name)
        ad.set("LAST-CHANGE", _now())
        typ = etree.SubElement(ad, _q("TYPE"))
        etree.SubElement(typ, _q("DATATYPE-DEFINITION-STRING-REF")).text = "DT-STRING"

    # spec-objects
    sobjs = etree.SubElement(rc, _q("SPEC-OBJECTS"))
    for i, (rid, text) in enumerate(_normalise(items), 1):
        rid = rid or f"REQ-{i:03d}"
        so = etree.SubElement(sobjs, _q("SPEC-OBJECT"))
        so.set("IDENTIFIER", f"OBJ-{i}")
        so.set("LAST-CHANGE", _now())
        values = etree.SubElement(so, _q("VALUES"))
        for the_value, ad_ref in ((str(rid), "AD-ID"), (str(text), "AD-TEXT")):
            av = etree.SubElement(values, _q("ATTRIBUTE-VALUE-STRING"))
            av.set("THE-VALUE", the_value)
            defn = etree.SubElement(av, _q("DEFINITION"))
            etree.SubElement(defn, _q("ATTRIBUTE-DEFINITION-STRING-REF")).text = ad_ref
        typ = etree.SubElement(so, _q("TYPE"))
        etree.SubElement(typ, _q("SPEC-OBJECT-TYPE-REF")).text = "ST-REQ"

    tree = etree.ElementTree(root)
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return path


def _reqif_xhtml_text(el) -> str:
    """Flatten a ReqIF XHTML attribute value to plain text.

    ReqIF stores rich text as an XHTML fragment under THE-VALUE; DOORS and
    Polarion use this for the requirement text itself. Block-level elements are
    separated by a space so "<p>a</p><p>b</p>" does not become "ab", and
    <br/> behaves the same way.
    """
    from lxml import etree
    parts = []
    for node in el.iter():
        tag = etree.QName(node).localname.lower() if node.tag is not etree.Comment else ""
        # the separator must precede the element's own text, otherwise the last
        # word of one block runs into the first word of the next
        if tag in ("p", "br", "div", "li", "tr", "td", "th"):
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


# ReqIF attribute values are typed; the text usually lives in STRING or XHTML,
# and everything else (enumerations, dates, numbers, flags) is metadata.
_REQIF_VALUE_TAGS = {
    "ATTRIBUTE-VALUE-STRING": "ATTRIBUTE-DEFINITION-STRING-REF",
    "ATTRIBUTE-VALUE-XHTML": "ATTRIBUTE-DEFINITION-XHTML-REF",
    "ATTRIBUTE-VALUE-ENUMERATION": "ATTRIBUTE-DEFINITION-ENUMERATION-REF",
    "ATTRIBUTE-VALUE-DATE": "ATTRIBUTE-DEFINITION-DATE-REF",
    "ATTRIBUTE-VALUE-INTEGER": "ATTRIBUTE-DEFINITION-INTEGER-REF",
    "ATTRIBUTE-VALUE-REAL": "ATTRIBUTE-DEFINITION-REAL-REF",
    "ATTRIBUTE-VALUE-BOOLEAN": "ATTRIBUTE-DEFINITION-BOOLEAN-REF",
}
_REQIF_DEFINITION_TAGS = ("ATTRIBUTE-DEFINITION-STRING", "ATTRIBUTE-DEFINITION-XHTML",
                          "ATTRIBUTE-DEFINITION-ENUMERATION", "ATTRIBUTE-DEFINITION-DATE",
                          "ATTRIBUTE-DEFINITION-INTEGER", "ATTRIBUTE-DEFINITION-REAL",
                          "ATTRIBUTE-DEFINITION-BOOLEAN")
# normalised long-names that identify the text / id attribute, best first
_REQIF_TEXT_NAMES = ("reqif.text", "text", "reqtext", "object_text", "requirement",
                     "requirement_text", "description")
_REQIF_ID_NAMES = ("reqif.foreignid", "reqid", "id", "object_identifier",
                   "requirement_id", "identifier", "absolute_number")


def read_reqif(path):
    """Read (id, text, metadata_dict) triples from a ReqIF file.

    Handles the attribute value types real tools emit — ``STRING`` **and**
    ``XHTML`` (DOORS/Polarion store the requirement text as XHTML), plus
    ENUMERATION / DATE / INTEGER / REAL / BOOLEAN — and recognises the usual
    long-names (``ReqIF.Text``, ``Object Text``, ``ReqIF.ForeignID``,
    ``Object Identifier``…). Every attribute that is not the id or the text is
    preserved as metadata, so rationale / applicability / verification columns
    survive the import.
    """
    from lxml import etree

    tree = etree.parse(path)

    # build ref -> normalised long-name map across every attribute-definition
    # type, and pick which definition holds the id and which holds the text.
    attr_names = {}                        # IDENTIFIER -> normalised long-name
    id_ref, text_ref = None, None
    id_rank, text_rank = len(_REQIF_ID_NAMES), len(_REQIF_TEXT_NAMES)

    for tag in _REQIF_DEFINITION_TAGS:
        for ad in tree.findall(f".//{_q(tag)}"):
            ident = ad.get("IDENTIFIER", "")
            norm = _col_norm(ad.get("LONG-NAME", ident))
            attr_names[ident] = norm
            if norm in _REQIF_ID_NAMES and _REQIF_ID_NAMES.index(norm) < id_rank:
                id_ref, id_rank = ident, _REQIF_ID_NAMES.index(norm)
            elif norm in _REQIF_TEXT_NAMES and _REQIF_TEXT_NAMES.index(norm) < text_rank:
                text_ref, text_rank = ident, _REQIF_TEXT_NAMES.index(norm)
    # reqgraph's own export uses these fixed identifiers
    id_ref = id_ref or ("AD-ID" if "AD-ID" in attr_names or not attr_names else "AD-ID")
    text_ref = text_ref or "AD-TEXT"

    out = []
    for so in tree.findall(f".//{_q('SPEC-OBJECT')}"):
        rid, text = None, None
        meta = {}
        for value_tag, ref_tag in _REQIF_VALUE_TAGS.items():
            for av in so.findall(f".//{_q(value_tag)}"):
                ref_el = av.find(f".//{_q(ref_tag)}")
                ref_val = (ref_el.text or "").strip() if ref_el is not None else ""
                if value_tag == "ATTRIBUTE-VALUE-XHTML":
                    holder = av.find(f"./{_q('THE-VALUE')}")
                    the_value = _reqif_xhtml_text(holder) if holder is not None else ""
                elif value_tag == "ATTRIBUTE-VALUE-ENUMERATION":
                    # values are references to ENUM-VALUE definitions
                    refs = av.findall(f".//{_q('ENUM-VALUE-REF')}")
                    the_value = ", ".join(
                        _reqif_enum_label(tree, (r.text or "").strip()) for r in refs)
                else:
                    the_value = (av.get("THE-VALUE") or "").strip()
                    if not the_value:      # some writers use a child element
                        child = av.find(f"./{_q('THE-VALUE')}")
                        if child is not None and child.text:
                            the_value = child.text.strip()
                if ref_val and ref_val == id_ref:
                    rid = the_value or rid
                elif ref_val and ref_val == text_ref:
                    text = the_value if text is None else text
                elif the_value:
                    name = attr_names.get(ref_val) or _col_norm(ref_val)
                    if name:
                        meta[name] = the_value
        if text is not None and text != "":
            out.append((rid, text, meta))
    return out


def _reqif_enum_label(tree, ref: str) -> str:
    """Resolve an ENUM-VALUE-REF to its human-readable LONG-NAME."""
    if not ref:
        return ""
    for ev in tree.findall(f".//{_q('ENUM-VALUE')}"):
        if ev.get("IDENTIFIER") == ref:
            return ev.get("LONG-NAME", ref)
    return ref
