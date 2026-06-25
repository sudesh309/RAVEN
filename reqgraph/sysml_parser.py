"""
reqgraph.sysml_parser
======================

Regex-based parser for SysML v2 textual notation (KerML/SysML v2 syntax).

Extracts named structural and behavioural elements from a SysML v2 model
and maps them to IREB semantic roles so they can be compared against
natural-language requirements via :mod:`reqgraph.sysml_compare`.

Role mapping
------------
+-------------------------------+-------------+-------------------------------+
| SysML keyword                 | Semantic    | Rationale                     |
|                               | Role        |                               |
+-------------------------------+-------------+-------------------------------+
| ``part def``, ``part``        | SUBJECT     | structural entity = "the      |
|                               |             | thing that shall…"            |
| ``action def``, ``action``    | PROCESS     | behaviour = the verb          |
| ``attribute def``,``attribute``| OBJECT     | property = the data acted on  |
| ``state def``, ``state``      | CONDITION   | mode = "when in state X"      |
| ``transition``                | CONDITION   | state change trigger          |
| ``port def``, ``port``        | ACTOR       | interaction point / interface |
| ``interface def``,``interface``| ACTOR      | interface definition          |
| ``require constraint``,       | CONSTRAINT  | formal assertion / bound      |
|   ``assume constraint``       |             |                               |
| ``connection def``,           | CONSTRAINT  | structural relationship        |
|   ``connection``              |             |                               |
| ``package``                   | *(skipped)* | namespace only                |
| ``requirement def``,          | *(special)* | extracted to                  |
|   ``requirement``             |             | ``SysMLModel.sysml_requirements``|
+-------------------------------+-------------+-------------------------------+

Usage::

    from reqgraph.sysml_parser import parse_sysml, read_sysml

    model = read_sysml("automotive.sysml")
    for e in model.elements:
        print(e.role.value, e.name)

    # Or parse text directly
    model = parse_sysml(open("automotive.sysml").read(), source_path="automotive.sysml")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import Role
from .errors import DataFormatError


# ---------------------------------------------------------------------------
# SysML keyword → semantic Role
# ---------------------------------------------------------------------------

_KEYWORD_ROLE: dict[str, Role | None] = {
    "part_def":           Role.SUBJECT,
    "part":               Role.SUBJECT,
    "action_def":         Role.PROCESS,
    "action":             Role.PROCESS,
    "attribute_def":      Role.OBJECT,
    "attribute":          Role.OBJECT,
    "state_def":          Role.CONDITION,
    "state":              Role.CONDITION,
    "transition":         Role.CONDITION,
    "port_def":           Role.ACTOR,
    "port":               Role.ACTOR,
    "interface_def":      Role.ACTOR,
    "interface":          Role.ACTOR,
    "require_constraint": Role.CONSTRAINT,
    "assume_constraint":  Role.CONSTRAINT,
    "connection_def":     Role.CONSTRAINT,
    "connection":         Role.CONSTRAINT,
    # namespace / requirement handled specially → None
    "package":            None,
    "requirement_def":    None,
    "requirement":        None,
}

# Keywords whose elements carry doc strings or text fields worth capturing
_REQ_KEYWORDS = {"requirement_def", "requirement"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SysMLElement:
    """One named element extracted from a SysML v2 model, tagged with its
    mapped semantic role for requirement comparison."""
    name: str           # extracted identifier, e.g. "BrakeSystem"
    role: Role          # mapped semantic role (SUBJECT / PROCESS / OBJECT / …)
    element_type: str   # canonical SysML keyword, e.g. "part_def", "action"
    doc: str = ""       # text from the nearest preceding doc /* … */ block
    package: str = ""   # enclosing package name (outermost wins)
    line: int = 0       # 1-based line number in source


@dataclass
class SysMLModel:
    """Parsed SysML v2 model: extracted elements + any embedded requirements."""
    elements: list[SysMLElement]
    source_path: str = ""
    packages: list[str] = field(default_factory=list)
    sysml_requirements: list[tuple] = field(default_factory=list)
    # ↑ list of (id, text) pairs from SysML ``requirement { … }`` blocks


# ---------------------------------------------------------------------------
# Internal regexes
# ---------------------------------------------------------------------------

# Extract doc /* … */ blocks (multi-line).  Captured text = group 1.
_DOC_RE = re.compile(r'doc\s*/\*(.*?)\*/', re.DOTALL | re.IGNORECASE)

# Extract text "…" fields inside requirement blocks.  Captured text = group 1.
_TEXT_FIELD_RE = re.compile(r'\btext\s+"([^"]+)"', re.DOTALL)

# Combined scanner: brace/close or a keyword declaration.
# Longer (multi-word) alternatives MUST come before shorter ones so "part def"
# never partially matches just "part".
_SCAN_RE = re.compile(
    r'(?P<open>\{)|(?P<close>\})|'
    r'\b(?P<kw>'
    r'part\s+def|action\s+def|state\s+def|attribute\s+def|'
    r'port\s+def|connection\s+def|requirement\s+def|interface\s+def|'
    r'require\s+constraint|assume\s+constraint|'
    r'part|action|state|attribute|port|connection|requirement|'
    r'transition|interface|package'
    r')\s+(?P<name>[A-Za-z_]\w*)',
    re.IGNORECASE,
)


def _canonicalise_kw(raw: str) -> str:
    """Lowercase and collapse internal whitespace to underscore."""
    return re.sub(r'\s+', '_', raw.strip().lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sysml(text: str, source_path: str = "") -> SysMLModel:
    """Parse a SysML v2 textual model and return a :class:`SysMLModel`.

    Raises :class:`~reqgraph.errors.DataFormatError` if the text is empty or
    contains no recognisable SysML declarations.
    """
    if not text or not text.strip():
        label = repr(source_path) if source_path else "input"
        raise DataFormatError(f"SysML model {label} is empty")

    # ------------------------------------------------------------------
    # Phase 1: extract doc comments and requirement text fields with their
    # positions in the *original* text (before stripping anything).
    # ------------------------------------------------------------------
    doc_end_to_text: dict[int, str] = {}
    for m in _DOC_RE.finditer(text):
        doc_end_to_text[m.end()] = m.group(1).strip()

    req_text_start_to_val: dict[int, str] = {}
    for m in _TEXT_FIELD_RE.finditer(text):
        req_text_start_to_val[m.start()] = m.group(1)

    # ------------------------------------------------------------------
    # Phase 2: build a "clean" copy with comments replaced by spaces so
    # the character positions stay in sync with the original.
    # Line comments: // … \n
    # Block comments: /* … */
    # Both are replaced with an equal-length run of spaces so regex
    # positions match the original text positions for line counting.
    # ------------------------------------------------------------------
    def _blank(m: re.Match) -> str:
        return ' ' * len(m.group())

    clean = re.sub(r'//[^\n]*', _blank, text)
    clean = re.sub(r'/\*.*?\*/', _blank, clean, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # Phase 3: single-pass scan on the clean copy — track brace depth and
    # package context, emit SysMLElement / sysml_requirements entries.
    # ------------------------------------------------------------------
    elements: list[SysMLElement] = []
    packages: list[str] = []
    sysml_requirements: list[tuple] = []

    # pkg_stack: list of [name, depth_when_entered]
    # depth_when_entered == depth value *after* the opening brace was counted
    pkg_stack: list[list] = []
    depth = 0

    for m in _SCAN_RE.finditer(clean):
        if m.group('open'):
            depth += 1
            continue

        if m.group('close'):
            depth -= 1
            # Pop any package whose scope has now closed.
            if pkg_stack and depth < pkg_stack[-1][1]:
                pkg_stack.pop()
            continue

        kw_raw = m.group('kw')
        name = m.group('name')
        kw = _canonicalise_kw(kw_raw)
        pkg = pkg_stack[-1][0] if pkg_stack else ""
        line_no = text[:m.start()].count('\n') + 1

        if kw == 'package':
            if name not in packages:
                packages.append(name)
            # depth + 1: the opening brace for this package hasn't been
            # seen yet; when it is, depth will be (current + 1).
            pkg_stack.append([name, depth + 1])
            continue

        if kw in _REQ_KEYWORDS:
            # Collect doc text and/or explicit text "…" field as the
            # semantic content of an embedded requirement.
            doc = _nearest_doc(doc_end_to_text, m.start())
            txt = _nearest_text_field(req_text_start_to_val, m.end(), gap=800)
            content = doc or txt
            if content:
                sysml_requirements.append((name, content))
            continue

        role = _KEYWORD_ROLE.get(kw)
        if role is None:
            continue

        doc = _nearest_doc(doc_end_to_text, m.start())
        elements.append(SysMLElement(
            name=name, role=role, element_type=kw,
            doc=doc, package=pkg, line=line_no,
        ))

    if not elements and not sysml_requirements:
        label = repr(source_path) if source_path else "input"
        raise DataFormatError(
            f"no SysML elements found in {label}; check that the content uses "
            f"SysML v2 textual notation (part def, action def, state def, …)")

    return SysMLModel(
        elements=elements,
        source_path=source_path,
        packages=packages,
        sysml_requirements=sysml_requirements,
    )


def read_sysml(path: str) -> SysMLModel:
    """Read a SysML v2 ``.sysml`` / ``.kerml`` file and return a :class:`SysMLModel`.

    Raises :class:`~reqgraph.errors.DataFormatError` for missing or
    unreadable files.
    """
    import os
    if not os.path.isfile(path):
        raise DataFormatError(f"SysML file not found: {path!r}")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise DataFormatError(f"could not read {path!r}: {exc}") from exc
    return parse_sysml(text, source_path=path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_doc(doc_map: dict[int, str], decl_start: int,
                 max_gap: int = 250) -> str:
    """Return the doc text whose block ended closest before *decl_start*.

    Only considers doc comments that end within *max_gap* characters before
    the declaration keyword.
    """
    best = ""
    best_dist = max_gap + 1
    for end_pos, doc_text in doc_map.items():
        dist = decl_start - end_pos
        if 0 < dist < best_dist:
            best = doc_text
            best_dist = dist
    return best


def _nearest_text_field(text_map: dict[int, str], decl_end: int,
                        gap: int = 800) -> str:
    """Return the first ``text "…"`` field that starts within *gap* characters
    after *decl_end* (i.e. inside the requirement's block body)."""
    best = ""
    best_dist = gap + 1
    for start_pos, val in text_map.items():
        dist = start_pos - decl_end
        if 0 < dist < best_dist:
            best = val
            best_dist = dist
    return best
