"""
Requirement cascade agent — checks the decomposition of the set
(system → subsystem → equipment) from the requirements' own trace links and
levels. No architecture model is needed.

Checks:

* structure: dangling parent links, cycles, orphans (lower-level requirements
  with no parent), level inversions and skipped levels;
* flow-down: non-leaf requirements that were never decomposed, children that
  only restate their parent;
* sufficiency: do the children, taken together, cover the parent's action,
  object and constraints (term coverage), and are the children's numeric bounds
  at least as tight as the parent's (budget consistency)?

An optional LLM pass judges whether each parent's children are *sufficient*
to satisfy it, returning candidate gaps.
"""

from __future__ import annotations

import re

from .base import LEVEL_KEYS, ReviewAgent, content_terms, meta_get, stem

_LEVEL_NUM = re.compile(r"(\d+)")
_GENERIC = {"perform", "provide", "capable", "capability", "function", "functions",
            "operation", "operations", "subsystem", "unit", "module", "component",
            "equipment", "drone", "aircraft", "vehicle", "maintain", "support", "level"}
_QTY = re.compile(
    r"(?P<cmp>not\s+exceed(?:ing)?|no\s+more\s+than|less\s+than|at\s+most|maximum(?:\s+of)?|"
    r"max\.?|within|up\s+to|under|below|<=?|at\s+least|no\s+less\s+than|minimum(?:\s+of)?|"
    r"min\.?|greater\s+than|more\s+than|above|exceed(?:ing)?|>=?)?\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|[a-zA-Z/°]{1,10})\b", re.I)
_UPPER = re.compile(r"not\s+exceed|no\s+more|less|at\s+most|max|within|up\s+to|under|below|<", re.I)
_LOWER = re.compile(r"at\s+least|no\s+less|min|greater|more\s+than|above|^exceed|>", re.I)


def _level(meta) -> int | None:
    v = meta_get(meta, *LEVEL_KEYS)
    m = _LEVEL_NUM.search(v) if v else None
    return int(m.group(1)) if m else None


def _quantities(text):
    out = []
    for m in _QTY.finditer(text or ""):
        cmp_ = (m.group("cmp") or "").lower()
        bound = "upper" if cmp_ and _UPPER.search(cmp_) else "lower" if cmp_ and _LOWER.search(cmp_) else ""
        ctx = text[max(0, m.start() - 40):m.start()]
        out.append({"value": float(m.group("val")), "unit": m.group("unit").lower(),
                    "bound": bound, "context": {stem(w) for w in content_terms(ctx)}})
    return out


def parent_links(ctx) -> dict:
    """child id -> [parent ids] as declared in metadata (cached on the context)."""
    def build():
        from ..traceability import _LINK_KEYS, _norm_id, _split_refs
        known = {_norm_id(r["id"]): r["id"] for r in ctx.requirements}
        links, dangling = {}, []
        for r in ctx.requirements:
            refs = []
            for key in _LINK_KEYS["parent"]:
                v = meta_get(r["meta"], key)
                if v:
                    refs += _split_refs(v)
            for ref in dict.fromkeys(refs):
                pid = known.get(_norm_id(ref))
                if pid is None:
                    dangling.append((r["id"], ref))
                elif pid != r["id"]:
                    links.setdefault(r["id"], []).append(pid)
        return links, dangling
    return ctx.cached("parent_links", build)


class CascadeAgent(ReviewAgent):
    name = "cascade"
    domain = "cascade"
    title = "Requirement cascade"
    description = ("Checks the decomposition tree built from parent / derived-from links "
                   "and levels: orphans, dangling links, cycles, level inversions, "
                   "undecomposed requirements, children that do not cover their parent's "
                   "action and constraints, and child budgets looser than the parent's.")
    llm_instruction = (
        "Each item is a parent requirement with its child requirements. Judge whether "
        "the children, taken together, are sufficient to satisfy the parent: name any "
        "aspect of the parent (function, condition, performance value, interface, mode) "
        "that no child refines, and any child that does not trace logically to the "
        "parent. Use the parent id as subject_id.")
    llm_batch_size = 6

    COVERAGE_MIN = 0.34

    def applicable(self, ctx):
        ok, why = super().applicable(ctx)
        if not ok:
            return ok, why
        links, dangling = parent_links(ctx)
        if links or dangling or any(_level(r["meta"]) is not None for r in ctx.requirements):
            return True, ""
        return False, ("no hierarchy information: add parent / derived_from links or a "
                       "level column to check the cascade")

    def analyze(self, ctx, result):
        from ..corpus import _lexical_similarity
        links, dangling = parent_links(ctx)
        ids = [r["id"] for r in ctx.requirements]
        levels = {r["id"]: _level(r["meta"]) for r in ctx.requirements}
        known_levels = sorted({l for l in levels.values() if l is not None})
        top = known_levels[0] if known_levels else None
        bottom = known_levels[-1] if known_levels else None
        children = {}
        for c, ps in links.items():
            for p in ps:
                children.setdefault(p, []).append(c)

        for c, ref in dangling:
            result.add("dangling_parent", "blocker", f"parent “{ref}” is not in the set", c,
                       hint="the parent was renamed, deleted or lives in another document")

        # cycles
        state, cyc = {}, set()

        def dfs(n, path):
            state[n] = 1
            for p in links.get(n, []):
                if state.get(p) == 1:
                    cyc.update(path[path.index(p):] if p in path else [n, p])
                elif not state.get(p):
                    dfs(p, path + [p])
            state[n] = 2
        for n in ids:
            if not state.get(n):
                dfs(n, [n])
        for n in sorted(cyc):
            result.add("cycle", "blocker", "part of a circular parent chain", n,
                       hint="a requirement cannot (indirectly) derive from itself")

        # structure per requirement
        for rid in ids:
            lv, ps = levels[rid], links.get(rid, [])
            if not ps and lv is not None and top is not None and lv > top:
                result.add("orphan", "major", f"level {lv} requirement with no parent", rid,
                           hint="link it to the higher-level requirement it refines, or mark it "
                                "as a new derived requirement with its own rationale")
            for p in ps:
                plv = levels.get(p)
                if lv is not None and plv is not None:
                    if lv <= plv:
                        result.add("level_inversion", "major",
                                   f"level {lv} but its parent {p} is level {plv}", rid,
                                   hint="a child must sit below its parent")
                    elif lv > plv + 1:
                        result.add("level_skip", "minor",
                                   f"skips from level {plv} ({p}) to level {lv}", rid,
                                   hint="confirm no intermediate-level requirement is missing")
            if lv is not None and bottom is not None and lv < bottom and not children.get(rid) \
                    and ctx.by_id[rid]["text"]:
                result.add("not_decomposed", "major",
                           f"level {lv} requirement has no child requirements", rid,
                           hint="flow it down, or record why it is verified at this level")

        # flow-down sufficiency
        coverages, budget_issues = [], 0
        for p, kids in children.items():
            ptext = ctx.by_id[p]["text"]
            kid_texts = [ctx.by_id[k]["text"] for k in kids]
            # the parent's core: what it does and to what (its action and object);
            # constraints are checked numerically below
            core = " ".join(filter(None, (ctx.role_text(p, "PROCESS"), ctx.role_text(p, "OBJECT"))))
            pterms = {stem(w) for w in content_terms(core or ptext, _GENERIC)}
            kterms = {stem(w) for t in kid_texts for w in content_terms(t)}
            if pterms:
                cov = len(pterms & kterms) / len(pterms)
                coverages.append(cov)
                missing = sorted(w for w in content_terms(core or ptext, _GENERIC)
                                 if stem(w) not in kterms)
                if cov < self.COVERAGE_MIN:
                    result.add("weak_flowdown", "minor",
                               f"children cover {cov:.0%} of the parent's action/object terms; "
                               f"not visibly refined: {', '.join(missing[:8])}", p,
                               evidence={"children": kids, "uncovered": missing[:20],
                                         "coverage": round(cov, 2)},
                               hint="add child requirements for the uncovered aspects",
                               status="candidate")
            if len(kids) == 1 and _lexical_similarity(ptext, kid_texts[0]) >= 0.85:
                result.add("restated_child", "minor",
                           f"{kids[0]} restates its parent without refining it", kids[0],
                           hint="a child should allocate or quantify, not copy")
            pq = _quantities(ptext)
            for k in kids:
                for cq in _quantities(ctx.by_id[k]["text"]):
                    for q in pq:
                        if q["unit"] != cq["unit"] or not q["bound"] or q["bound"] != cq["bound"]:
                            continue
                        if not (q["context"] & cq["context"]):
                            continue
                        looser = (q["bound"] == "upper" and cq["value"] > q["value"]) or \
                                 (q["bound"] == "lower" and cq["value"] < q["value"])
                        if looser:
                            budget_issues += 1
                            result.add("budget_violation", "major",
                                       f"{q['bound']} bound {cq['value']:g} {cq['unit']} is looser "
                                       f"than parent {p}'s {q['value']:g} {q['unit']}", k,
                                       evidence={"parent": p}, status="candidate",
                                       hint="a child bound must be at least as tight as the parent's")

        # tree for UIs / reports
        roots = [r for r in ids if not links.get(r)]
        depth = {}

        def walk(n, d, seen):
            if n in seen:
                return
            depth[n] = max(depth.get(n, 0), d)
            for k in children.get(n, []):
                walk(k, d + 1, seen | {n})
        for r in roots:
            walk(r, 0, frozenset())
        result.data["tree"] = {"roots": roots, "children": children}
        n = len(ids)
        n_children = sum(len(v) for v in children.values())
        result.metrics = {
            "roots": len(roots), "links": n_children, "depth": max(depth.values(), default=0) + 1,
            "levels": known_levels,
            "pct_linked": round(100.0 * len(links) / n, 1) if n else 0.0,
            "mean_child_coverage": round(sum(coverages) / len(coverages), 2) if coverages else None,
            "budget_violations": budget_issues}

    def llm_payload(self, ctx, result):
        children = result.data.get("tree", {}).get("children", {})
        return [{"parent": {"id": p, "text": ctx.by_id[p]["text"]},
                 "children": [{"id": k, "text": ctx.by_id[k]["text"]} for k in kids]}
                for p, kids in children.items()]
