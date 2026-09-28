"""
reqgraph.pluscal
================

Requirement → **PlusCal / TLA+** formalisation.

A parsed requirement already carries everything a formal model needs: *who*
acts (SUBJECT), *under which circumstances* (CONDITION + its EARS marker),
*how binding* it is (MODALITY), *what* happens (PROCESS + OBJECT, possibly a
compound AND/OR), and *how fast* (a CONSTRAINT such as "within 4 seconds").
This module turns that structure into a PlusCal algorithm inside a TLA+ module,
plus a TLC model configuration, so the requirement can be model-checked:

* the **SUBJECT** becomes a (fair) process that performs the action;
* each **CONDITION** becomes a Boolean variable — raised by an ``Environment``
  process for event-style triggers (WHEN / IF / AS SOON AS …), or an arbitrary
  but fixed initial value for state-style ones (WHILE / DURING / WHERE …);
* each **action** (PROCESS + OBJECT) becomes a Boolean "has happened" flag;
* the **MODALITY** decides the obligation: *shall / must* → the properties are
  checked; *should* → checked but advisory; *may* → nothing to check;
  *shall not* → a prohibition (safety) property;
* a **deadline** ("within N units") adds a discrete clock using Lamport's
  real-time idiom (time may not advance past a pending upper bound) and a
  ``DeadlineMet`` invariant.

The output is a **reference model**: TLC confirms it satisfies its own
properties (the requirement is consistent and satisfiable as formalised). Its
value is that the *properties* (``Response``, ``DeadlineMet``, ``Prohibition``…)
are the requirement stated formally; when the reference process is replaced by
a design, TLC checks the design against them. Verified behaviour: a design that
responds too slowly violates ``DeadlineMet``; one that never responds violates
``Response``.

Every generated identifier is traced back to the requirement element it came
from (``PlusCalSpec.mapping``), and anything the translation cannot formalise
(a rate, an accuracy, a non-temporal bound) is listed in ``notes`` instead of
being silently dropped.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

from .core import RequirementGraph, Rel, Role
from .errors import ReqGraphError
from .templates import RUPP_TEMPLATE, Template

# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

# TLA+ / PlusCal reserved words, standard-module operators and the names the
# translator introduces -- a slug equal to one of these gets a suffix.
_RESERVED = {
    # TLA+
    "assume", "assumption", "axiom", "case", "choose", "constant", "constants",
    "domain", "else", "enabled", "except", "extends", "if", "in", "instance",
    "let", "local", "module", "other", "sf_", "subset", "then", "theorem",
    "unchanged", "union", "variable", "variables", "wf_", "true", "false",
    "boolean", "string", "nat", "int", "seq",
    # PlusCal
    "algorithm", "assert", "await", "begin", "call", "define", "do", "either",
    "elsif", "end", "fair", "goto", "macro", "or", "print", "procedure",
    "process", "return", "skip", "when", "while", "with",
    # introduced by the translator / this generator
    "self", "pc", "stack", "vars", "procset", "init", "next", "spec",
    "termination", "clock",
}
_ARTICLES = {"the", "a", "an"}
_ASCII_MAP = {"≤": "<=", "≥": ">=", "°": " deg", "µ": "u", "–": "-", "—": "-",
              "‘": "'", "’": "'", "“": '"', "”": '"', "×": "x", "±": "+/-"}


def _ascii(text: str) -> str:
    """ASCII-only rendering for TLA+ comments and identifiers."""
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    return (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore").decode("ascii"))


def _slug(text: str, max_len: int = 44) -> str:
    """lowercase_underscore identifier from free text ("14,000" → "14000")."""
    s = _ascii(text).lower()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # thousands separators
    s = re.sub(r"(?<=\d)\.(?=\d)", "_", s)             # 4.5 -> 4_5
    words = [w for w in re.findall(r"[a-z0-9]+", s) if w not in _ARTICLES]
    out = ""
    for w in words:
        nxt = f"{out}_{w}" if out else w
        if len(nxt) > max_len and out:
            break
        out = nxt
    if not out:
        return ""
    if out[0].isdigit():
        out = "v_" + out
    return out


def _camel(slug: str) -> str:
    return "".join(p.capitalize() for p in slug.split("_") if p) or "System"


def _module_name(req_id: Optional[str]) -> str:
    if not req_id:
        return "Requirement"
    name = re.sub(r"[^A-Za-z0-9_]+", "_", _ascii(str(req_id))).strip("_")
    if not name or not re.search(r"[A-Za-z]", name):
        name = f"REQ_{name}" if name else "Requirement"
    if name[0].isdigit():
        name = "R_" + name
    return name


class _Names:
    """Hands out unique, non-reserved identifiers."""

    def __init__(self, taken=()):
        self.used = {t.lower() for t in taken}

    def take(self, base: str, fallback: str) -> str:
        base = base or fallback
        cand = base
        if cand.lower() in _RESERVED:
            cand = f"{cand}_v"
        i = 2
        while cand.lower() in self.used:
            cand = f"{base}_{i}"
            i += 1
        self.used.add(cand.lower())
        return cand


# ---------------------------------------------------------------------------
# requirement analysis
# ---------------------------------------------------------------------------

# how each condition marker behaves in the model
_EVENT_MARKERS = ("when", "whenever", "once", "after", "as soon as", "upon",
                  "on", "if", "in case", "in the event", "in the event that")
_STATE_MARKERS = ("while", "during", "as long as", "where", "provided",
                  "provided that", "given", "given that", "unless", "except when")
_NEGATED_MARKERS = ("unless", "except when")

_DEADLINE_RE = re.compile(
    r"\b(?:within|in\s+less\s+than|in\s+under|in\s+at\s+most|no\s+later\s+than|"
    r"not\s+later\s+than|not\s+more\s+than|at\s+most|in)\s+"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|microseconds?|us|seconds?|secs?|s|"
    r"minutes?|mins?|hours?|hrs?|h|days?|cycles?|ticks?|frames?|steps?)\b",
    re.I)
_UNIT_NAMES = {
    "ms": "milliseconds", "msec": "milliseconds", "msecs": "milliseconds",
    "millisecond": "milliseconds", "milliseconds": "milliseconds",
    "us": "microseconds", "microsecond": "microseconds", "microseconds": "microseconds",
    "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds",
    "seconds": "seconds", "min": "minutes", "mins": "minutes", "minute": "minutes",
    "minutes": "minutes", "h": "hours", "hr": "hours", "hrs": "hours",
    "hour": "hours", "hours": "hours", "day": "days", "days": "days",
}
_MAX_COMFORTABLE_DEADLINE = 5000       # TLC state space grows linearly with it


@dataclass
class _Condition:
    text: str            # original text, e.g. "When the cabin altitude exceeds…"
    marker: str          # "when"
    style: str           # "event" | "state"
    negated: bool        # "unless X" means the requirement applies while ~X
    var: str = ""


@dataclass
class _Action:
    text: str            # "deploy the passenger oxygen masks"
    process: str
    obj: str
    var: str = ""


def _strip_marker(text: str, marker: str) -> str:
    t = text.strip().rstrip(",").strip()
    if marker and t.lower().startswith(marker.lower()):
        t = t[len(marker):]
    t = re.sub(r"^[\s,]+", "", t)
    t = re.sub(r"[\s,]*\bthen\s*$", "", t, flags=re.I)   # "If X, then" → "X"
    return t.strip()


def _conditions(g: RequirementGraph) -> list[_Condition]:
    out = []
    for n in g.by_role(Role.CONDITION):
        text = n.text.strip()
        if not text:
            continue
        marker = (n.attrs.get("marker") or "").lower().strip()
        if not marker:
            first = text.split()[0].lower() if text.split() else ""
            marker = first
        style = "state" if marker in _STATE_MARKERS else "event"
        out.append(_Condition(text=text, marker=marker, style=style,
                              negated=marker in _NEGATED_MARKERS))
    return out


def _children(g: RequirementGraph, node_id: str, rel: Rel) -> list:
    return [g.nodes[e.target] for e in g.edges
            if e.source == node_id and e.rel is rel and e.target in g.nodes]


def _actions(g: RequirementGraph) -> tuple[list[_Action], str]:
    """The action(s) and how they combine: 'single' | 'AND' | 'OR'."""
    root = next((n for n in g.nodes.values() if n.role is Role.ROOT), None)
    heads = _children(g, root.id, Rel.HAS_ACTION) if root else []
    combine = "single"
    action_nodes = []
    for h in heads:
        if h.role is Role.OPERATOR:
            combine = (h.attrs.get("operator") or "AND").upper()
            action_nodes.extend(n for n in _children(g, h.id, Rel.OPERAND)
                                if n.role is Role.ACTION)
        elif h.role is Role.ACTION:
            action_nodes.append(h)

    acts = []
    for a in action_nodes:
        procs = _children(g, a.id, Rel.HAS_PROCESS)
        objs = _children(g, a.id, Rel.ACTS_ON)
        dets = _children(g, a.id, Rel.HAS_DETAILS)
        p = " ".join(n.text.strip() for n in procs).strip()
        o = " ".join(n.text.strip() for n in objs + dets).strip()
        if p:
            acts.append(_Action(text=f"{p} {o}".strip(), process=p, obj=o))

    if not acts:                        # graph without ACTION groups
        procs = g.by_role(Role.PROCESS)
        objs = g.by_role(Role.OBJECT)
        for i, p in enumerate(procs):
            o = objs[i].text.strip() if i < len(objs) else ""
            acts.append(_Action(text=f"{p.text.strip()} {o}".strip(),
                                process=p.text.strip(), obj=o))
    if combine not in ("AND", "OR"):
        combine = "single" if len(acts) <= 1 else "AND"
    if len(acts) <= 1:
        combine = "single"
    return acts, combine


def _obligation(g: RequirementGraph) -> tuple[str, str]:
    """('mandatory' | 'advisory' | 'permitted' | 'prohibited', modality text)."""
    mods = g.by_role(Role.MODALITY)
    text = mods[0].text.strip() if mods else ""
    low = text.lower()
    if re.search(r"\bnot\b|cannot|can't|shan't|mustn't|won't", low):
        return "prohibited", text
    if re.search(r"\b(shall|must|will|is required to|are required to|has to|have to)\b", low):
        return "mandatory", text
    if re.search(r"\bshould\b", low):
        return "advisory", text
    if re.search(r"\b(may|can|could|might)\b", low):
        return "permitted", text
    return "mandatory", text      # an unrecognised custom-template modality


def _deadline(g: RequirementGraph, text: str):
    """(value:int, unit_label:str, source_text) for an upper time bound, or None."""
    sources = [n.text for n in g.by_role(Role.CONSTRAINT)] + [text]
    for src in sources:
        m = _DEADLINE_RE.search(src)
        if not m:
            continue
        raw = m.group("value").replace(",", ".")
        unit = _UNIT_NAMES.get(m.group("unit").lower(), m.group("unit").lower())
        value = float(raw)
        scale = 0
        while scale < 3 and abs(value * 10 ** scale - round(value * 10 ** scale)) > 1e-9:
            scale += 1
        ticks = int(round(value * 10 ** scale))
        label = unit if scale == 0 else f"{10 ** -scale:g} {unit}"
        if ticks <= 0:
            continue
        return ticks, label, m.group(0)
    return None


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class PlusCalSpec:
    """A generated PlusCal/TLA+ module and its TLC model configuration."""
    module: str
    tla: str
    cfg: str
    requirement: str
    pattern: str                 # event | unwanted | state | optional | ubiquitous
    obligation: str              # mandatory | advisory | permitted | prohibited
    properties: list = field(default_factory=list)   # [{name, kind, formula, meaning, checked}]
    mapping: list = field(default_factory=list)      # [{role, text, spec, detail}]
    notes: list = field(default_factory=list)
    constants: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, directory: str) -> tuple[str, str]:
        """Write ``<module>.tla`` and ``<module>.cfg``; returns both paths."""
        os.makedirs(directory, exist_ok=True)
        tla_path = os.path.join(directory, f"{self.module}.tla")
        cfg_path = os.path.join(directory, f"{self.module}.cfg")
        with open(tla_path, "w", encoding="utf-8") as fh:
            fh.write(self.tla)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(self.cfg)
        return tla_path, cfg_path


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _comment_lines(text: str, width: int = 74) -> list[str]:
    words, lines, cur = _ascii(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return [f"\\* {ln}" for ln in lines] or ["\\*"]


def requirement_to_pluscal(graph_or_text, *, req_id: Optional[str] = None,
                           template: Template = RUPP_TEMPLATE,
                           extractor=None) -> PlusCalSpec:
    """Formalise one requirement as a PlusCal algorithm + TLC configuration.

    Accepts a parsed :class:`RequirementGraph` or the requirement text. Raises
    :class:`ReqGraphError` when the requirement states no action at all (there is
    nothing to formalise — the completeness check flags the same defect).
    """
    from .quality import classify_ears

    if isinstance(graph_or_text, RequirementGraph):
        g = graph_or_text
    else:
        from .parser import RequirementParser
        g = RequirementParser(template, extractor).parse(str(graph_or_text))
    text = g.generate().strip()
    req_id = req_id or g.metadata.get("id")

    # Without a modality the obligation is undefined -- formalising it as
    # "mandatory" would invent a commitment the text never makes.
    if not g.by_role(Role.MODALITY):
        raise ReqGraphError(
            "cannot formalise: the requirement has no modality (shall / should / "
            "may / shall not), so what it obliges is undefined.")
    conds = _conditions(g)
    acts, combine = _actions(g)
    if not acts:
        raise ReqGraphError(
            "cannot formalise: the requirement states no action (no PROCESS "
            "element was found). Add the verb the subject must perform.")
    obligation, modality_text = _obligation(g)
    ears = classify_ears(g)
    pattern = ("event" if ears.startswith("event") else
               "unwanted" if ears.startswith("unwanted") else
               "state" if ears.startswith("state") else
               "optional" if ears.startswith("optional") else "ubiquitous")
    deadline = _deadline(g, text) if obligation in ("mandatory", "advisory") else None

    module = _module_name(req_id)
    subj_nodes = g.by_role(Role.SUBJECT) or g.by_role(Role.ACTOR)
    subject_text = subj_nodes[0].text.strip() if subj_nodes else "the system"

    # --- names ---------------------------------------------------------------
    define_names = {"TypeOK", "Trigger", "Done", "Pending", "Response",
                    "DeadlineMet", "Prohibition", "Never", "Deadline", "MaxTime"}
    names = _Names(taken={n.lower() for n in define_names} | {module.lower()})
    for i, c in enumerate(conds, 1):
        c.var = names.take(_slug(_strip_marker(c.text, c.marker)), f"condition_{i}")
    for i, a in enumerate(acts, 1):
        a.var = names.take(_slug(a.text), f"action_{i}")
    proc_slug = _slug(subject_text) or "system"
    proc_name = _camel(proc_slug)
    labels = {"Raise", "Wait", "Act", "Tick", "Start", "Idle", "Environment", "Clock"}
    if proc_name in define_names or proc_name in labels or proc_name.lower() in _RESERVED:
        proc_name += "Proc"
    names.take(proc_name, "SystemProc")

    event_conds = [c for c in conds if c.style == "event"]
    trig_terms = [(f"~{c.var}" if c.negated else c.var) for c in conds]
    trigger = " /\\ ".join(trig_terms) if trig_terms else "TRUE"
    joiner = " \\/ " if combine == "OR" else " /\\ "
    done = joiner.join(a.var for a in acts)
    if len(acts) > 1:
        done = f"({done})"

    # --- variables -----------------------------------------------------------
    var_lines = []
    for c in conds:
        init = "FALSE" if c.style == "event" else "\\in BOOLEAN"
        op = "=" if c.style == "event" else ""
        decl = f"{c.var} {op} {init}".replace("  ", " ")
        how = ("raised by the environment" if c.style == "event"
               else "arbitrary, fixed for the run")
        var_lines.append((decl, f"CONDITION ({c.marker or '?'}): {how}"))
    for a in acts:
        var_lines.append((f"{a.var} = FALSE", f"ACTION has happened: {a.text}"))
    if deadline:
        var_lines.append(("clock = 0", "time elapsed since the trigger"))

    # --- define block --------------------------------------------------------
    bool_vars = [c.var for c in conds] + [a.var for a in acts]
    type_terms = [f"{v} \\in BOOLEAN" for v in bool_vars]
    if deadline:
        type_terms += ["clock \\in 0..MaxTime"]
    defines = []
    defines.append("TypeOK == " + ("\n              /\\ ".join(type_terms)
                                   if len(type_terms) == 1 else
                                   "/\\ " + "\n              /\\ ".join(type_terms)))
    defines.append(f"Trigger == {trigger}")
    defines.append(f"Done == {done}")

    properties = [{"name": "TypeOK", "kind": "invariant", "formula": "types",
                   "meaning": "every variable stays within its declared type",
                   "checked": True}]
    invariants, temporal = ["TypeOK"], []

    if obligation in ("mandatory", "advisory"):
        if trigger == "TRUE":
            defines.append("Response == <>Done")
            meaning = "the action eventually happens"
        else:
            defines.append("Response == Trigger ~> Done")
            meaning = "whenever the trigger holds, the action eventually happens"
        properties.append({"name": "Response", "kind": "liveness",
                           "formula": defines[-1].split("==", 1)[1].strip(),
                           "meaning": meaning, "checked": True})
        temporal.append("Response")
        if deadline:
            defines.append("Pending == Trigger /\\ ~Done")
            defines.append("DeadlineMet == Pending => clock <= Deadline")
            properties.append({
                "name": "DeadlineMet", "kind": "invariant",
                "formula": "Pending => clock <= Deadline",
                "meaning": f"the action happens within {deadline[0]} {deadline[1]} "
                           f"of the trigger", "checked": True})
            invariants.append("DeadlineMet")
    elif obligation == "prohibited":
        if trigger == "TRUE":
            defines.append("Never == ~Done")
            properties.append({"name": "Never", "kind": "invariant", "formula": "~Done",
                               "meaning": "the prohibited action never happens",
                               "checked": True})
            invariants.append("Never")
        else:
            av = [a.var for a in acts]
            fired = " \\/ ".join(f"(~{v} /\\ {v}')" for v in av)
            tup = ", ".join(dict.fromkeys(bool_vars))
            defines.append(f"Prohibition == [][~(Trigger /\\ ({fired}))]_<<{tup}>>")
            properties.append({
                "name": "Prohibition", "kind": "action",
                "formula": f"[][~(Trigger /\\ ({fired}))]_<<{tup}>>",
                "meaning": "the prohibited action is never taken while the "
                           "condition holds", "checked": True})
            temporal.append("Prohibition")
    else:  # permitted
        properties.append({"name": "(none)", "kind": "permission", "formula": "",
                           "meaning": "a permission ('may') creates no obligation, so "
                                      "there is nothing for TLC to check beyond typing",
                           "checked": False})

    # --- processes -----------------------------------------------------------
    procs = []
    if event_conds:
        body = []
        for i, c in enumerate(event_conds, 1):
            lab = f"Raise{i}" if len(event_conds) > 1 else "Raise"
            body.append(f"  {lab}:\n"
                        f"    either {c.var} := TRUE;\n"
                        f"    or skip;\n"
                        f"    end either;")
        procs.append('process Environment = "environment"\n'
                     "begin\n" + "\n".join(body) + "\nend process;")

    sys_body = []
    fair = "fair "
    if obligation in ("mandatory", "advisory"):
        if trigger != "TRUE":
            sys_body.append("  Wait:\n    await Trigger;")
        if combine == "OR":
            alts = "\n    or ".join(f"{a.var} := TRUE;" for a in acts)
            sys_body.append(f"  Act:\n    either {alts}\n    end either;")
        else:
            for i, a in enumerate(acts, 1):
                lab = f"Act{i}" if len(acts) > 1 else "Act"
                sys_body.append(f"  {lab}:\n    {a.var} := TRUE;")
    elif obligation == "permitted":
        fair = ""
        guard = "await Trigger; " if trigger != "TRUE" else ""
        alts = "\n    or ".join(f"{guard}{a.var} := TRUE;" for a in acts)
        sys_body.append(f"  Act:\n    either {alts}\n    or skip;\n    end either;")
    else:  # prohibited
        fair = ""
        if trigger == "TRUE":
            sys_body.append("  Idle:\n    skip;   \\* a compliant system never takes the action")
        else:
            alts = "\n    or ".join(f"await ~Trigger; {a.var} := TRUE;" for a in acts)
            sys_body.append(f"  Act:\n    either {alts}\n    or skip;\n    end either;")
    procs.append(f'{fair}process {proc_name} = "{proc_slug}"\n'
                 "begin\n" + "\n".join(sys_body) + "\nend process;")

    if deadline:
        procs.append('process Clock = "clock"\n'
                     "begin\n"
                     "  Start:\n"
                     "    await Trigger;   \\* a bounded response is timed from its trigger\n"
                     "  Tick:\n"
                     "    while ~Done /\\ clock < MaxTime do\n"
                     "      \\* time may not pass a pending upper bound (RTBound idiom)\n"
                     "      await ~(Pending /\\ clock >= Deadline);\n"
                     "      clock := clock + 1;\n"
                     "    end while;\n"
                     "end process;")

    # --- assemble module -------------------------------------------------------
    bar = "-" * max(4, (74 - len(module) - 8) // 2)
    head = [f"{bar} MODULE {module} {bar}"]
    head.append(f"\\* Generated by RAVEN from requirement {req_id or '(no id)'}:")
    head += _comment_lines(text)
    head.append(f"\\* Pattern: {ears}. Obligation: {obligation}"
                f" ('{_ascii(modality_text) or '-'}').")
    head.append("EXTENDS Naturals, TLC")
    head.append("")
    if deadline:
        amount = (f"{deadline[0]} x {deadline[1]}" if deadline[1][0].isdigit()
                  else f"{deadline[0]} {deadline[1]}")
        head.append(f"CONSTANTS Deadline,  \\* {amount} "
                    f"(from \"{_ascii(deadline[2])}\")")
        head.append("          MaxTime    \\* bound on explored time (Deadline + 1)")
        head.append("")

    alg = [f"(* --algorithm {module}", "variables"]
    for i, (decl, why) in enumerate(var_lines):
        sep = ";" if i == len(var_lines) - 1 else ","
        alg.append(f"    {decl}{sep}   \\* {_ascii(why)}")
    alg.append("")
    alg.append("define")
    for d in defines:
        alg.append("    " + d)
    alg.append("end define;")
    alg.append("")
    for p in procs:
        alg.append(p)
        alg.append("")
    alg.append("end algorithm; *)")

    tla = "\n".join(head + alg) + "\n" + "=" * 77 + "\n"

    constants = {}
    cfg = [f"\\* TLC model for {module} -- generated by RAVEN"]
    if deadline:
        constants = {"Deadline": deadline[0], "MaxTime": deadline[0] + 1}
        cfg.append("CONSTANTS")
        cfg += [f"    {k} = {v}" for k, v in constants.items()]
    cfg.append("SPECIFICATION Spec")
    cfg.append("INVARIANTS " + " ".join(invariants))
    if temporal:
        cfg.append("PROPERTIES " + " ".join(temporal))
    cfg.append("\\* a reactive requirement may legitimately wait forever for its trigger")
    cfg.append("CHECK_DEADLOCK FALSE")
    cfg_text = "\n".join(cfg) + "\n"

    # --- traceability map + notes -------------------------------------------
    mapping = []
    for c in conds:
        mapping.append({"role": "CONDITION", "text": c.text, "spec": c.var,
                        "detail": ("event trigger raised by Environment"
                                   if c.style == "event" else
                                   "state (arbitrary initial value)")
                                  + (" -- negated ('unless')" if c.negated else "")})
    mapping.append({"role": "SUBJECT", "text": subject_text,
                    "spec": f"process {proc_name}",
                    "detail": "fair process (obligation)" if fair else
                              "unfair process (no obligation to act)"})
    mapping.append({"role": "MODALITY", "text": modality_text or "(none)",
                    "spec": ", ".join(p["name"] for p in properties if p["checked"]),
                    "detail": obligation})
    for a in acts:
        mapping.append({"role": "PROCESS+OBJECT", "text": a.text, "spec": a.var,
                        "detail": "action flag" + (f" ({combine})" if combine != "single" else "")})
    notes = []
    # a formal model of an incomplete requirement inherits the gap -- say so
    from .quality import check_completeness
    comp = check_completeness(g)
    for f in comp["missing"]:
        if f["severity"] in ("blocker", "major"):
            notes.append(f"Requirement is incomplete ({f['label']}): the model "
                         f"formalises it as written -- {f['hint']}.")
    if deadline:
        mapping.append({"role": "CONSTRAINT", "text": deadline[2],
                        "spec": f"Deadline = {deadline[0]}",
                        "detail": f"1 clock tick = 1 {deadline[1].rstrip('s')}"
                                  if not deadline[1][0].isdigit() else
                                  f"1 clock tick = {deadline[1]}"})
        if deadline[0] > _MAX_COMFORTABLE_DEADLINE:
            notes.append(f"Deadline of {deadline[0]} ticks makes the explored state "
                         f"space large; consider a coarser time unit.")
    for n in g.by_role(Role.CONSTRAINT):
        if not deadline or deadline[2] not in n.text:
            notes.append(f"CONSTRAINT not formalised: \"{n.text.strip()}\" -- it is not "
                         f"an upper time bound; add an invariant over a model variable "
                         f"if it must be checked.")
    if obligation == "advisory":
        notes.append("'should' is advisory: the properties are checked, but a "
                     "violation is a missed recommendation rather than a failed "
                     "obligation.")
    if obligation == "permitted":
        notes.append("'may' grants a permission; TLC can only check what must or "
                     "must not happen, so no property is generated.")
    if pattern == "state" and obligation in ("mandatory", "advisory"):
        notes.append("State-driven condition modelled as fixed for the run; if it "
                     "can change while the action is pending, extend Environment "
                     "to toggle it and refine Response accordingly.")

    return PlusCalSpec(module=module, tla=tla, cfg=cfg_text, requirement=text,
                       pattern=pattern, obligation=obligation,
                       properties=properties, mapping=mapping, notes=notes,
                       constants=constants)


def requirements_to_pluscal(items, *, template: Template = RUPP_TEMPLATE,
                            extractor=None) -> list:
    """Formalise a set of ``(id, text, meta)`` items. Requirements that cannot be
    formalised are returned as ``(req_id, error_message)`` tuples instead of
    aborting the whole batch."""
    from .parser import RequirementParser
    parser = RequirementParser(template, extractor)
    out = []
    for i, it in enumerate(items, 1):
        if isinstance(it, (tuple, list)):
            rid, txt = it[0], it[1]
        else:
            rid, txt = None, it
        rid = rid or f"R{i}"
        try:
            out.append(requirement_to_pluscal(parser.parse(txt), req_id=rid))
        except ReqGraphError as exc:
            out.append((rid, str(exc)))
    return out


# ---------------------------------------------------------------------------
# optional validation with the TLA+ tools (PlusCal translator + TLC)
# ---------------------------------------------------------------------------

_JAR_CANDIDATES = ("tla2tools.jar", os.path.join("tools", "tla2tools.jar"),
                   os.path.expanduser("~/tla2tools.jar"),
                   os.path.expanduser("~/.tlaplus/tla2tools.jar"),
                   "/usr/share/java/tla2tools.jar", "/opt/tla/tla2tools.jar")


def find_tla_tools(jar: Optional[str] = None) -> Optional[str]:
    """Locate ``tla2tools.jar`` (argument, ``$RAVEN_TLA2TOOLS``, common paths)."""
    for cand in (jar, os.environ.get("RAVEN_TLA2TOOLS"), *_JAR_CANDIDATES):
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def tla_tools_status(jar: Optional[str] = None) -> dict:
    path = find_tla_tools(jar)
    java = shutil.which("java")
    return {"available": bool(path and java), "jar": path, "java": java,
            "detail": ("ready" if path and java else
                       "install Java and download tla2tools.jar from "
                       "https://github.com/tlaplus/tlaplus/releases, then set "
                       "RAVEN_TLA2TOOLS=/path/to/tla2tools.jar"
                       if not path else "Java runtime not found on PATH")}


def _clean(out: str) -> str:
    return "\n".join(ln for ln in out.splitlines()
                     if not ln.startswith(("Picked up JAVA_TOOL_OPTIONS",
                                           "Picked up _JAVA_OPTIONS")))


def validate_spec(spec, jar: Optional[str] = None, *, cfg: Optional[str] = None,
                  timeout: int = 180) -> dict:
    """Translate the PlusCal with ``pcal.trans`` and model-check it with TLC.

    ``spec`` is a :class:`PlusCalSpec` or raw module text (then pass ``cfg``).
    Returns a dict with ``available``, ``translated``, ``checked``, ``ok``,
    ``violations`` (``[{kind, name}]``), state counts, the counterexample text
    when there is one, and ``error`` for tool/parse failures. Never raises for
    a spec problem -- a broken candidate is a result, not an exception.
    """
    status = tla_tools_status(jar)
    result = {"available": status["available"], "translated": False,
              "checked": False, "ok": False, "violations": [],
              "states_generated": None, "distinct_states": None,
              "counterexample": "", "error": None, "output": ""}
    if not status["available"]:
        result["error"] = status["detail"]
        return result

    if isinstance(spec, PlusCalSpec):
        tla, cfg_text, module = spec.tla, spec.cfg, spec.module
    else:
        tla, cfg_text = str(spec), cfg or ""
        m = re.search(r"-{4,}\s*MODULE\s+(\w+)", tla)
        if not m:
            result["error"] = "no '---- MODULE <name> ----' header found"
            return result
        module = m.group(1)
    if not cfg_text.strip():
        result["error"] = "no TLC configuration supplied"
        return result

    java = status["java"]
    with tempfile.TemporaryDirectory(prefix="raven_tla_") as tmp:
        with open(os.path.join(tmp, f"{module}.tla"), "w", encoding="utf-8") as fh:
            fh.write(tla)
        with open(os.path.join(tmp, f"{module}.cfg"), "w", encoding="utf-8") as fh:
            fh.write(cfg_text)
        try:
            tr = subprocess.run(
                [java, "-cp", status["jar"], "pcal.trans", "-nocfg", f"{module}.tla"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            result["error"] = "PlusCal translation timed out"
            return result
        tr_out = _clean(tr.stdout + tr.stderr)
        if tr.returncode != 0 or "Translation completed" not in tr_out:
            result["error"] = "PlusCal translation failed"
            result["output"] = tr_out[-3000:]
            return result
        result["translated"] = True
        try:
            tlc = subprocess.run(
                [java, "-XX:+UseParallelGC", "-cp", status["jar"], "tlc2.TLC",
                 "-workers", "1", "-nowarning", "-metadir",
                 os.path.join(tmp, "states"), module],
                cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            result["error"] = f"TLC did not finish within {timeout}s (state space too large?)"
            return result
    out = _clean(tlc.stdout + tlc.stderr)
    result["output"] = out[-4000:]
    m = re.search(r"([\d,]+) states generated, ([\d,]+) distinct states found", out)
    if m:
        result["states_generated"] = int(m.group(1).replace(",", ""))
        result["distinct_states"] = int(m.group(2).replace(",", ""))
    for m in re.finditer(r"Invariant (\w+) is violated", out):
        result["violations"].append({"kind": "invariant", "name": m.group(1)})
    for m in re.finditer(r"Temporal propert(?:y|ies) (\w+)?\s*(?:was|were) violated", out):
        result["violations"].append({"kind": "temporal", "name": m.group(1) or "?"})
    for m in re.finditer(r"Action property (\w+) of the specification is violated|"
                         r"Action property (\w+) is violated", out):
        result["violations"].append({"kind": "action", "name": m.group(1) or m.group(2)})
    if "Deadlock reached" in out:
        result["violations"].append({"kind": "deadlock", "name": "deadlock"})
    if result["violations"]:
        idx = out.find("Error:")
        result["counterexample"] = out[idx: idx + 3000] if idx >= 0 else ""
    if "Model checking completed. No error has been found." in out:
        result["checked"] = True
        result["ok"] = not result["violations"]
    elif result["violations"]:
        result["checked"] = True
    else:
        # parse / semantic error in the translated module, or TLC failure
        err = re.search(r"(\*\*\*.*?Error.*?\n.*?\n|Semantic errors:.*?\n.*?\n|"
                        r"Error: .*?\n)", out, re.S)
        result["error"] = (err.group(0).strip() if err else "TLC did not complete")
    return result
