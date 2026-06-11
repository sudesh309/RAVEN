"""
reqgraph.io_formats
===================

Batch import / export of requirement sets:

* CSV / Excel  (pandas + openpyxl) -- a flat table with one decomposed column
  per requirement element, plus type / EARS / quality columns.
* ReqIF        (lxml) -- the OMG Requirements Interchange Format used by DOORS,
  Polarion, etc. A minimal, well-formed subset (ID + text) is written/read;
  export->import round-trips the requirement text.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable, Optional

from .core import Role
from .errors import DataFormatError
from .parser import RequirementParser
from .templates import RUPP_TEMPLATE, Template

logger = logging.getLogger(__name__)

_ELEMENT_COLUMNS = [Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
                    Role.PROCESS, Role.OBJECT, Role.CONSTRAINT]


def _normalise(items):
    """Accept ['text', ...] or [(id, text), ...]; yield (id, text)."""
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

    Fault isolation: a row that fails to parse is recorded with its error
    message instead of aborting the whole batch."""
    import pandas as pd
    from .quality import enrich

    parser = RequirementParser(template, extractor)
    rows = []
    for rid, text in _normalise(items):
        row = {"id": rid, "text": text, "type": None, "ears_pattern": None,
               "roundtrip_ok": False, "error": ""}
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


def _rows_from_df(df, text_column, id_column):
    import pandas as pd
    if text_column not in df.columns:
        raise DataFormatError(
            f"text column {text_column!r} not found; available columns: "
            f"{list(df.columns)}")
    items = []
    for _, r in df.iterrows():
        txt = r[text_column]
        if pd.isna(txt) or not str(txt).strip():
            continue                      # skip blank rows quietly
        rid = r[id_column] if id_column in df.columns else None
        if rid is not None and pd.isna(rid):
            rid = None
        items.append((None if rid is None else str(rid), str(txt)))
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
    """Read (id, text) pairs back from a ReqIF file written by ``write_reqif``."""
    from lxml import etree

    tree = etree.parse(path)
    out = []
    for so in tree.findall(f".//{_q('SPEC-OBJECT')}"):
        rid, text = None, None
        for av in so.findall(f".//{_q('ATTRIBUTE-VALUE-STRING')}"):
            ref = av.find(f".//{_q('ATTRIBUTE-DEFINITION-STRING-REF')}")
            ref_val = ref.text if ref is not None else ""
            if ref_val == "AD-ID":
                rid = av.get("THE-VALUE")
            elif ref_val == "AD-TEXT":
                text = av.get("THE-VALUE")
        if text is not None:
            out.append((rid, text))
    return out
