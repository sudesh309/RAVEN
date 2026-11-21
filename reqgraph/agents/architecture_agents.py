"""
Architecture team — runs only when an architecture model (SysML v2 / v1) is
supplied. Kept separate from the quality team on purpose: the quality verdict
on a requirement set never depends on whether a model happens to be loaded.

* :class:`AllocationAgent` (requirements → architecture): every requirement is
  realised by an element of the architecture, explicitly (``satisfy`` /
  ``allocate`` / requirement ``subject`` / an allocation column) or at least
  recognisably (its subject names an element).
* :class:`ArchitectureJustificationAgent` (architecture → requirements): every
  part, interface and function exists because some requirement needs it —
  otherwise it is gold-plating or a requirement is missing.
* :class:`DerivedRequirementsAgent` (architecture-driven cascade): requirements
  written inside the model are in the specification, decomposed elements carry
  the derived requirements their decomposition implies, and child requirements
  are allocated inside their parent's branch of the architecture.
"""

from __future__ import annotations

import re

from .base import ALLOCATION_KEYS, LEVEL_KEYS, ReviewAgent, content_terms, meta_get, split_camel, stem
from .cascade import _level, parent_links

_GENERIC_ARCH = {"subsystem", "system", "unit", "module", "component", "assembly",
                 "equipment", "def", "block", "part", "the"}
_PART_KINDS = ("part_def",)
_IFACE_KINDS = ("port_def", "interface_def")
_FUNC_KINDS = ("action_def",)
SEMANTIC_MIN = 0.6
TEXT_MATCH_MIN = 0.75


def _stems(text, generic=_GENERIC_ARCH):
    return {stem(w) for w in content_terms(text, generic)}


def _name_score(phrase: str, element_name: str) -> float:
    from ..corpus import _lexical_similarity
    a = _stems(phrase)
    b = _stems(split_camel(element_name))
    if not a or not b:
        return 0.0
    jac = len(a & b) / len(a | b)
    # containment either way: "barometric pressure sensor" names Barometer,
    # "motor" names BrushlessMotor
    contain = max(len(a & b) / len(a), len(a & b) / len(b))
    seq = _lexical_similarity(" ".join(sorted(a)), " ".join(sorted(b))) if jac >= 0.5 else 0.0
    return round(max(0.5 * jac + 0.5 * contain, seq), 3)


def _norm(s):
    return re.sub(r"[\s_\-]+", "", (s or "").strip().upper())


def allocations(ctx) -> dict:
    """rid -> {"explicit": [def], "semantic": [[def, score]], "unresolved": [ref],
    "model_requirement": name|None}. Cached on the context."""
    def build():
        from ..corpus import _lexical_similarity
        m = ctx.architecture
        defs = [d for d in m.definitions() if d.kind in _PART_KINDS + _IFACE_KINDS]
        part_defs = [d for d in defs if d.kind in _PART_KINDS]
        depth = m.depth()
        roots = [n for n, d in depth.items() if d == 0]
        usage_type = {e.name: m.resolve(e.type) for e in m.elements
                      if not e.kind.endswith("_def") and e.type}
        out = {}
        for r in ctx.requirements:
            rid = r["id"]
            entry = {"explicit": [], "semantic": [], "unresolved": [], "model_requirement": None}
            # 1. the same requirement written in the model?
            mreq = None
            for q in m.requirements:
                if _norm(q.short_id) == _norm(rid) or _norm(q.name) == _norm(rid):
                    mreq = q
                    break
            if mreq is None and r["text"]:
                best = max(((q, _lexical_similarity(q.text, r["text"])) for q in m.requirements
                            if q.text), key=lambda t: t[1], default=(None, 0))
                if best[1] >= TEXT_MATCH_MIN:
                    mreq = best[0]
            names = {_norm(rid)}
            if mreq is not None:
                entry["model_requirement"] = mreq.name
                names |= {_norm(mreq.name), _norm(mreq.short_id)} - {""}
            # 2. explicit links
            refs = []
            for key in ALLOCATION_KEYS:
                v = meta_get(r["meta"], key)
                if v:
                    from ..traceability import _split_refs
                    refs += _split_refs(v)
            for l in m.links:
                if l.kind in ("satisfy", "subject") and _norm(re.split(r"::|\.", l.target)[-1]) in names:
                    refs.append(l.source)
                elif l.kind == "allocate" and _norm(re.split(r"::|\.", l.source)[-1]) in names:
                    refs.append(l.target)
            for ref in dict.fromkeys(refs):
                d = m.resolve(ref) or usage_type.get(ref)
                if d:
                    if d not in entry["explicit"]:
                        entry["explicit"].append(d)
                else:
                    entry["unresolved"].append(ref)
            # 3. semantic: the requirement's subject names an element
            if not entry["explicit"] and r["text"]:
                subj = ctx.role_text(rid, "SUBJECT") or r["text"].split(" shall")[0]
                if not _stems(subj) and roots:
                    entry["semantic"] = [[roots[0], SEMANTIC_MIN]]
                else:
                    scored = []
                    for d in part_defs:
                        s = _name_score(subj, d.name)
                        for u, t in usage_type.items():
                            if t == d.name:
                                s = max(s, _name_score(subj, u))
                        if s >= SEMANTIC_MIN:
                            scored.append([d.name, s])
                    scored.sort(key=lambda t: (-t[1], depth.get(t[0], 99)))
                    if scored:
                        entry["semantic"] = [t for t in scored if t[1] >= scored[0][1] - 0.05][:3]
            out[rid] = entry
        return out
    return ctx.cached("allocations", build)


def _allocated_to(entry) -> list:
    return entry["explicit"] or [d for d, _ in entry["semantic"]]


def _ancestors(model, name) -> set:
    seen, stack = set(), [name]
    while stack:
        for p in model.parents(stack.pop()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


# ---------------------------------------------------------------------------
class AllocationAgent(ReviewAgent):
    name = "allocation"
    domain = "architecture"
    title = "Requirement allocation (req → architecture)"
    description = ("Checks every requirement is realised by an architecture element: "
                   "explicit satisfy / allocate / subject links or allocation columns, "
                   "otherwise an element its subject names; flags requirements no element "
                   "realises and links to elements that do not exist.")
    requires_architecture = True
    llm_instruction = (
        "Each item is a requirement with no confirmed allocation and the list of "
        "architecture elements. Propose the element that should satisfy it, using the "
        "element name as subject_id and the requirement id in the message; say why.")

    def analyze(self, ctx, result):
        alloc = allocations(ctx)
        n_exp = n_sem = 0
        for r in ctx.requirements:
            e = alloc[r["id"]]
            for ref in e["unresolved"]:
                result.add("unknown_allocation_target", "blocker",
                           f"allocated to “{ref}”, which is not in the architecture",
                           r["id"], hint="fix the reference or add the element to the model")
            if e["explicit"]:
                n_exp += 1
            elif e["semantic"]:
                n_sem += 1
                d, s = e["semantic"][0]
                result.add("implicit_allocation", "minor",
                           f"no explicit allocation; its subject matches {d} ({s:.0%})", r["id"],
                           evidence={"candidates": e["semantic"]}, status="candidate",
                           hint=f"add 'satisfy {r['id']} by …' or an allocated_to entry for {d}")
            elif r["text"]:
                result.add("unallocated", "major", "no architecture element realises this requirement",
                           r["id"], hint="allocate it to an element, or the architecture is "
                                         "missing the element that delivers it")
        n = len(ctx.requirements)
        result.data["allocations"] = alloc
        result.metrics = {"pct_explicit": round(100.0 * n_exp / n, 1) if n else 0.0,
                          "pct_allocated": round(100.0 * (n_exp + n_sem) / n, 1) if n else 0.0,
                          "unallocated": n - n_exp - n_sem}

    def llm_payload(self, ctx, result):
        open_ids = {f.subject_id for f in result.findings if f.rule == "unallocated"}
        if not open_ids:
            return []
        elems = [{"name": d.name, "doc": d.doc[:160]} for d in ctx.architecture.definitions()
                 if d.kind in _PART_KINDS]
        return [{"requirement": {"id": r["id"], "text": r["text"]}, "elements": elems}
                for r in ctx.requirements if r["id"] in open_ids]


# ---------------------------------------------------------------------------
class ArchitectureJustificationAgent(ReviewAgent):
    name = "architecture_justification"
    domain = "architecture"
    title = "Architecture justification (architecture → req)"
    description = ("Checks every part, interface and function in the model is driven by "
                   "a requirement (on itself, its parents or its sub-parts). Unjustified "
                   "elements are gold-plating or reveal a missing requirement.")
    requires_architecture = True
    llm_instruction = (
        "Each item is an architecture element no requirement justifies. Say whether it "
        "looks like gold-plating or a missing requirement, and draft the missing "
        "requirement in the hint. Use the element name as subject_id.")

    def analyze(self, ctx, result):
        m = ctx.architecture
        alloc = allocations(ctx)
        allocated = {d for e in alloc.values() for d in _allocated_to(e)}
        # requirements written inside the model justify their subjects / satisfiers too
        for l in m.links:
            if l.kind in ("satisfy", "subject", "allocate"):
                d = m.resolve(l.source if l.kind != "allocate" else l.target)
                if d:
                    allocated.add(d)
        req_stems = [_stems(r["text"]) for r in ctx.requirements]
        n_checked = n_unjust = 0
        for d in m.definitions():
            if d.kind in _PART_KINDS:
                n_checked += 1
                if d.name in allocated:
                    continue
                down = m.descendants(d.name) & allocated
                up = _ancestors(m, d.name) & allocated
                if not down and not up:
                    n_unjust += 1
                    result.add("unjustified_element", "major",
                               "no requirement is allocated to this part, its parents or its "
                               "sub-parts", d.name, subject_kind="element",
                               hint="gold-plating, or a requirement is missing for it")
            elif d.kind in _IFACE_KINDS + _FUNC_KINDS:
                n_checked += 1
                es = _stems(split_camel(re.sub(r"(Port|Interface|Def)$", "", d.name)))
                hit = d.name in allocated or any(es and len(es & rs) / len(es) >= 0.66
                                                for rs in req_stems)
                if not hit:
                    n_unjust += 1
                    kind = "interface" if d.kind in _IFACE_KINDS else "function"
                    result.add(f"unspecified_{kind}", "minor",
                               f"no requirement specifies this {kind}", d.name,
                               subject_kind="element",
                               hint=f"add an {kind} requirement, or remove it from the model")
        result.metrics = {"elements_checked": n_checked, "unjustified": n_unjust,
                          "pct_justified": round(100.0 * (n_checked - n_unjust) / n_checked, 1)
                          if n_checked else 0.0}

    def llm_payload(self, ctx, result):
        m = ctx.architecture
        docs = {d.name: d.doc for d in m.definitions()}
        return [{"element": f.subject_id, "doc": docs.get(f.subject_id, "")[:200],
                 "parents": m.parents(f.subject_id)}
                for f in result.findings if f.subject_kind == "element"][:30]


# ---------------------------------------------------------------------------
class DerivedRequirementsAgent(ReviewAgent):
    name = "derived_requirements"
    domain = "architecture"
    title = "Derived requirements & allocation cascade"
    description = ("Checks requirements written in the model appear in the specification, "
                   "each decomposed element carries the derived requirements its "
                   "decomposition implies, child requirements are allocated within their "
                   "parent's architecture branch, and requirement levels match "
                   "architecture depth.")
    requires_architecture = True
    llm_instruction = (
        "Each item is an architecture element that was introduced by decomposing a "
        "parent element but has no requirement of its own. Draft the derived "
        "requirement it needs (\"The <element> shall …\", measurable) in the hint, "
        "consistent with the parent requirements given. Use the element name as subject_id.")

    def analyze(self, ctx, result):
        from ..corpus import _lexical_similarity
        m = ctx.architecture
        alloc = allocations(ctx)
        allocated = {}
        for rid, e in alloc.items():
            for d in _allocated_to(e):
                allocated.setdefault(d, []).append(rid)

        # 1. requirements captured in the model but missing from the specification
        matched = {e["model_requirement"] for e in alloc.values() if e["model_requirement"]}
        for q in m.requirements:
            if q.name in matched or q.is_def and not q.text:
                continue
            best = max((_lexical_similarity(q.text, r["text"]) for r in ctx.requirements
                        if q.text and r["text"]), default=0)
            result.add("model_requirement_missing", "major",
                       f"requirement “{q.name}” exists in the model but not in the "
                       f"specification", q.name, subject_kind="element",
                       evidence={"text": q.text, "subjects": q.subjects, "best_match": round(best, 2)},
                       hint="add it to the requirement set (with rationale and parent) or "
                            "remove it from the model")

        # 2. decomposition implies derived requirements
        gaps = {}
        for d, rids in allocated.items():
            for c in m.children(d):
                if c in allocated or (m.descendants(c) & set(allocated)):
                    continue
                gaps.setdefault(c, set()).update(rids)
        kinds = {x.name: x.kind for x in m.definitions()}
        for c, rids in sorted(gaps.items()):
            iface = kinds.get(c) in _IFACE_KINDS
            result.add("derived_requirement_expected", "minor" if iface else "major",
                       f"decomposed from an element with requirements, but no "
                       f"{'interface ' if iface else ''}requirement is allocated to it",
                       c, subject_kind="element",
                       evidence={"parent_elements": m.parents(c), "parent_requirements": sorted(rids)},
                       hint="write the derived requirement(s) that flow down to it and link "
                            "them to " + ", ".join(sorted(rids)[:4]))
        result.data["derived_gaps"] = {c: sorted(r) for c, r in gaps.items()}

        # 3. child requirements allocated inside the parent's branch
        links, _ = parent_links(ctx)
        n_cons = n_checked = 0
        for child, parents in links.items():
            c_el = set(_allocated_to(alloc.get(child, {"explicit": [], "semantic": []})))
            for p in parents:
                p_el = set(_allocated_to(alloc.get(p, {"explicit": [], "semantic": []})))
                if not c_el or not p_el:
                    continue
                n_checked += 1
                branch = set(p_el)
                for x in p_el:
                    branch |= m.descendants(x)
                if c_el & branch:
                    n_cons += 1
                    continue
                result.add("allocation_outside_parent_branch", "major",
                           f"allocated to {', '.join(sorted(c_el))}, outside its parent {p}'s "
                           f"element {', '.join(sorted(p_el))}", child,
                           evidence={"parent": p, "child_elements": sorted(c_el),
                                     "parent_elements": sorted(p_el)},
                           status="confirmed" if alloc[child]["explicit"] and alloc[p]["explicit"]
                           else "candidate",
                           hint="re-trace the child to the right parent, or re-allocate it")

        # 4. requirement level vs architecture depth
        depth = m.depth()
        lv = {r["id"]: _level(r["meta"]) for r in ctx.requirements}
        known = [v for v in lv.values() if v is not None]
        if known:
            top = min(known)
            for rid, e in alloc.items():
                if lv.get(rid) is None or not e["explicit"]:
                    continue
                ds = [depth[d] for d in e["explicit"] if d in depth]
                if ds and min(abs((lv[rid] - top) - x) for x in ds) >= 2:
                    result.add("level_depth_mismatch", "minor",
                               f"level {lv[rid]} requirement allocated at architecture depth "
                               f"{min(ds)}", rid,
                               hint="check the requirement level or the allocation")
        result.metrics = {"model_requirements": len(m.requirements),
                          "model_requirements_missing": sum(f.rule == "model_requirement_missing"
                                                            for f in result.findings),
                          "derived_gaps": len(gaps),
                          "cascade_pairs_checked": n_checked,
                          "pct_cascade_consistent": round(100.0 * n_cons / n_checked, 1)
                          if n_checked else None}

    def llm_payload(self, ctx, result):
        m = ctx.architecture
        docs = {d.name: d.doc for d in m.definitions()}
        return [{"element": c, "doc": docs.get(c, "")[:200], "parent_elements": m.parents(c),
                 "parent_requirements": [{"id": r, "text": ctx.by_id[r]["text"]} for r in rids[:4]]}
                for c, rids in result.data.get("derived_gaps", {}).items()][:24]
