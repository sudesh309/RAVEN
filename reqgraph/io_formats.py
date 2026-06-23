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

    Extra columns beyond id/text are normalised (lowercase, spaces→underscore)
    and returned in the metadata dict. NaN values are omitted from the dict.
    """
    import pandas as pd
    if text_column not in df.columns:
        raise DataFormatError(
            f"text column {text_column!r} not found; available columns: "
            f"{list(df.columns)}")
    extra_cols = [c for c in df.columns if c not in (text_column, id_column)]
    norm_map = {c: _col_norm(c) for c in extra_cols}   # original → normalised

    items = []
    for _, r in df.iterrows():
        txt = r[text_column]
        if pd.isna(txt) or not str(txt).strip():
            continue                      # skip blank rows quietly
        rid = r[id_column] if id_column in df.columns else None
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


def read_reqif(path):
    """Read (id, text, metadata_dict) triples from a ReqIF file.

    Reads ALL ATTRIBUTE-VALUE-STRING elements, not just the standard
    AD-ID / AD-TEXT pair, so extra attributes (rationale, applicability, etc.)
    added by tools like DOORS or Polarion are preserved on import.
    """
    from lxml import etree

    tree = etree.parse(path)

    # build ref -> normalised_name map from SPEC-OBJECT-TYPE SPEC-ATTRIBUTES
    attr_names = {}   # IDENTIFIER -> normalised long-name
    id_ref = "AD-ID"
    text_ref = "AD-TEXT"

    for ad in tree.findall(f".//{_q('ATTRIBUTE-DEFINITION-STRING')}"):
        ident = ad.get("IDENTIFIER", "")
        long_name = ad.get("LONG-NAME", ident)
        norm = _col_norm(long_name)
        attr_names[ident] = norm
        # detect which ref is the id and which is the text
        if norm in ("reqid", "id") or ident == "AD-ID":
            id_ref = ident
        elif norm in ("reqtext", "text") or ident == "AD-TEXT":
            text_ref = ident

    out = []
    for so in tree.findall(f".//{_q('SPEC-OBJECT')}"):
        rid, text = None, None
        meta = {}
        for av in so.findall(f".//{_q('ATTRIBUTE-VALUE-STRING')}"):
            ref_el = av.find(f".//{_q('ATTRIBUTE-DEFINITION-STRING-REF')}")
            ref_val = ref_el.text if ref_el is not None else ""
            the_value = av.get("THE-VALUE", "")
            if ref_val == id_ref:
                rid = the_value
            elif ref_val == text_ref:
                text = the_value
            else:
                norm_name = attr_names.get(ref_val, _col_norm(ref_val))
                if norm_name and the_value:
                    meta[norm_name] = the_value
        if text is not None:
            out.append((rid, text, meta))
    return out
