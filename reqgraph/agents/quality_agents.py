"""
Requirement quality team — works on the requirement set alone (no model).

* :class:`WritingQualityAgent` — INCOSE GtWR wording rules (ambiguity, atomicity,
  modality, passive voice, vague quantifiers).
* :class:`CompletenessAgent` — each requirement fully stated (INCOSE C4/C7) and
  the set covers the requirement types a specification is expected to have.
* :class:`RationaleAgent` — every requirement carries a real rationale /
  justification, a source, and an adequate verification method.
* :class:`ConsistencyAgent` — duplicate ids, near-duplicate statements and
  conflicting statements about the same subject.
"""

from __future__ import annotations

import re

from .base import (RATIONALE_KEYS, SOURCE_KEYS, VERIFICATION_KEYS, ReviewAgent,
                   content_terms, meta_get, stem)


# ---------------------------------------------------------------------------
class WritingQualityAgent(ReviewAgent):
    name = "writing_quality"
    domain = "quality"
    title = "Writing quality"
    description = ("Checks each requirement against INCOSE GtWR wording rules: binding "
                   "modality, weak/ambiguous words, vague quantifiers, passive voice, "
                   "non-atomic and compound statements.")
    llm_instruction = (
        "For each requirement, identify ambiguity a rule checker misses: undefined "
        "terms, unbounded ranges, implicit assumptions, multiple interpretations, "
        "design solutions stated as requirements. Ignore anything listed in "
        "'already_found'.")

    _RULES = (("missing_modality", "blocker", "no binding modality (shall/must)",
               "state the obligation with 'shall'"),
              ("non_atomic", "major", "not atomic (several conjunctions)",
               "split into one requirement per verifiable statement"),
              ("compound_requirement", "major", "compound requirement (several modal clauses)",
               "one 'shall' per requirement"),
              ("vague_quantifier", "major", "vague quantifier",
               "replace 'some/several/many/various' with a number"),
              ("passive_voice", "minor", "passive voice hides the responsible actor",
               "name the system or actor that performs the action"))

    def analyze(self, ctx, result):
        from ..quality import check_quality, quality_score
        scores = []
        for r in ctx.requirements:
            if not r["text"]:
                result.add("empty_text", "blocker", "requirement has no text", r["id"])
                scores.append(0)
                continue
            q = check_quality(r["text"])
            score, _ = quality_score(q)
            scores.append(score)
            for key, sev, msg, hint in self._RULES:
                if q.get(key):
                    result.add(key, sev, msg, r["id"], hint=hint)
            if q.get("weak_words"):
                result.add("weak_words", "major",
                           "ambiguous / weak words: " + ", ".join(q["weak_words"]),
                           r["id"], hint="replace with measurable, verifiable terms",
                           evidence={"words": q["weak_words"]})
        n = len(scores)
        result.metrics = {"mean_score": round(sum(scores) / n, 1) if n else 0.0,
                          "pct_clean": round(100.0 * sum(s == 100 for s in scores) / n, 1) if n else 0.0}


# ---------------------------------------------------------------------------
class CompletenessAgent(ReviewAgent):
    name = "completeness"
    domain = "quality"
    title = "Completeness"
    description = ("Each requirement is fully stated (subject, action, binding modality, "
                   "no TBD/TBC, measurable quantities with units and tolerances, no "
                   "dangling references) and the set covers functional, performance, "
                   "interface, safety and off-nominal behaviour.")
    llm_instruction = (
        "Judge whether each requirement is complete enough to design and verify: "
        "missing operating conditions or modes, missing performance bounds, missing "
        "failure / off-nominal behaviour, undefined interfaces. Ignore items in "
        "'already_found'.")

    EXPECTED_TYPES = ("functional", "performance", "interface", "safety")

    def analyze(self, ctx, result):
        from ..quality import check_completeness, classify_ears, classify_type
        scores, types, ears = [], {}, {}
        for r in ctx.requirements:
            if not r["text"]:
                continue
            g = ctx.graph(r["id"])
            c = check_completeness(g)
            scores.append(c["score"])
            for f in c["missing"]:
                result.add(f["key"], f["severity"], f["label"], r["id"], hint=f["hint"])
            t = classify_type(r["text"])
            types[t] = types.get(t, 0) + 1
            try:
                e = classify_ears(g)
            except Exception:
                e = "ubiquitous"
            ears[e] = ears.get(e, 0) + 1
        n = len(scores)
        if n >= 5:
            for t in self.EXPECTED_TYPES:
                if not types.get(t):
                    result.add(f"no_{t}_requirements", "minor" if t != "safety" else "major",
                               f"the set has no {t} requirements",
                               hint=f"confirm {t} needs are captured elsewhere or add them")
            if not any(k.startswith("unwanted") for k in ears):
                result.add("no_off_nominal", "major",
                           "no off-nominal (IF … THEN) requirements: failure and abnormal "
                           "behaviour is unspecified",
                           hint="add EARS unwanted-behaviour requirements for each failure "
                                "condition from the FHA / hazard log")
        result.metrics = {"mean_score": round(sum(scores) / n, 1) if n else 0.0,
                          "pct_complete": round(100.0 * sum(s >= 80 for s in scores) / n, 1) if n else 0.0,
                          "types": types, "ears_patterns": ears}


# ---------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"^\s*(tbd|tbc|tbr|n/?a|none|-+|\?+|todo|see above|same as above|"
                          r"self[- ]explanatory|obvious|required|needed|as required)\s*\.?\s*$", re.I)
_WHY_CUE = re.compile(
    r"\b(to|so that|in order|because|since|ensure|ensures|enable|enables|allow|allows|"
    r"prevent|prevents|avoid|protect|mitigat\w*|comply|compliance|required by|mandated|"
    r"regulation|standard|cs-?\d+|far|easa|faa|do-?\d+|arp\d+|iso|hazard|safety|derived|"
    r"allocated|flow(?:s|ed)? down|customer|stakeholder|mission|need|trade|analysis|"
    r"budget|margin|risk)\b", re.I)
_VM_WORDS = {"test": "T", "t": "T", "analysis": "A", "a": "A", "inspection": "I", "i": "I",
             "demonstration": "D", "demo": "D", "d": "D", "review": "I", "similarity": "A",
             "simulation": "A", "sim": "A"}


class RationaleAgent(ReviewAgent):
    name = "rationale"
    domain = "quality"
    title = "Rationale, source & verification"
    description = ("Every requirement states why it exists (rationale / justification), "
                   "where it comes from (source or parent), and how it will be verified "
                   "(Test / Analysis / Inspection / Demonstration) with a method that "
                   "suits the wording.")
    llm_instruction = (
        "For each item judge whether the 'rationale' genuinely justifies the "
        "requirement (explains the need, the source, the derivation or the chosen "
        "value) rather than restating it, and whether the verification method is "
        "adequate. Report weak rationales with a concrete improvement hint.")

    def analyze(self, ctx, result):
        from ..corpus import _lexical_similarity
        from ..quality import classify_type
        from ..traceability import _LINK_KEYS, assign_verification_method
        n = len(ctx.requirements)
        have_rat = have_src = have_vm = 0
        for r in ctx.requirements:
            rid, text, meta = r["id"], r["text"], r["meta"]
            rat = meta_get(meta, *RATIONALE_KEYS)
            src = meta_get(meta, *SOURCE_KEYS) or meta_get(meta, *_LINK_KEYS["parent"])
            vm = meta_get(meta, *VERIFICATION_KEYS)
            safety = classify_type(text) == "safety" if text else False
            if not rat:
                result.add("missing_rationale", "blocker" if safety else "major",
                           "no rationale / justification" + (" on a safety requirement" if safety else ""),
                           rid, hint="state why the requirement exists and where its values come from")
            else:
                have_rat += 1
                if _PLACEHOLDER.match(rat):
                    result.add("trivial_rationale", "major",
                               f"rationale is a placeholder: \u201c{rat[:80]}\u201d",
                               rid, hint="explain the need, source document, or analysis behind it")
                elif _lexical_similarity(rat, text) >= 0.6:
                    result.add("restating_rationale", "major",
                               "rationale restates the requirement instead of justifying it", rid,
                               hint="say why — the need, hazard, regulation or analysis")
                elif len(content_terms(rat)) < 3:
                    result.add("trivial_rationale", "major",
                               f"rationale is too thin to justify anything: \u201c{rat[:80]}\u201d",
                               rid, hint="explain the need, source document, or analysis behind it")
                elif not _WHY_CUE.search(rat):
                    result.add("rationale_without_reason", "info",
                               "rationale does not cite a purpose, source or analysis", rid,
                               hint="link it to a stakeholder need, regulation, hazard or trade study")
                if re.search(r"\d", text) and not re.search(r"\d|analysis|budget|margin|derived|"
                                                             r"standard|regulation|study", rat, re.I):
                    result.add("unjustified_value", "info",
                               "the requirement states a value but the rationale does not say "
                               "where it comes from", rid,
                               hint="cite the analysis, budget or standard the value is derived from")
            if src:
                have_src += 1
            else:
                result.add("missing_source", "minor", "no source or parent requirement recorded",
                           rid, hint="record the originating need, document or parent requirement")
            if not vm:
                result.add("missing_verification_method", "major", "no verification method",
                           rid, hint="assign Test, Analysis, Inspection or Demonstration",
                           evidence={"suggested": assign_verification_method(text).value})
            else:
                have_vm += 1
                codes = {_VM_WORDS.get(w) for w in re.findall(r"[a-z]+", vm.lower())} - {None}
                if not codes:
                    result.add("unknown_verification_method", "minor",
                               f"verification method “{vm}” is not T/A/I/D", rid,
                               hint="use Test, Analysis, Inspection or Demonstration")
                else:
                    sug = assign_verification_method(text).value
                    sug_code = _VM_WORDS.get(sug.lower())
                    if sug_code == "T" and codes == {"I"}:
                        result.add("weak_verification_method", "minor",
                                   "measurable performance verified by Inspection only", rid,
                                   hint="a quantified requirement usually needs Test or Analysis",
                                   evidence={"declared": vm, "suggested": sug})
        result.metrics = {
            "pct_with_rationale": round(100.0 * have_rat / n, 1) if n else 0.0,
            "pct_with_source": round(100.0 * have_src / n, 1) if n else 0.0,
            "pct_with_verification": round(100.0 * have_vm / n, 1) if n else 0.0}


# ---------------------------------------------------------------------------
_NEG = re.compile(r"\b(shall|must|will)\s+not\b|\bshall\s+never\b|\bcannot\b|\bprohibit", re.I)
_QTY = re.compile(r"(\d+(?:\.\d+)?)\s*(%|[a-zA-Z/°]{1,10})")


class ConsistencyAgent(ReviewAgent):
    name = "consistency"
    domain = "quality"
    title = "Consistency"
    description = ("Finds duplicate identifiers, near-duplicate requirements, and "
                   "conflicting statements (same subject and action with opposite "
                   "polarity or different values).")
    llm_instruction = (
        "Look across the items for logical conflicts or inconsistent terminology "
        "(the same thing named differently, contradictory values or modes). Only "
        "report pairs you can name; put both ids in the message.")
    llm_batch_size = 40

    DUP, SIMILAR = 0.88, 0.75

    def analyze(self, ctx, result):
        from ..corpus import _lexical_similarity
        seen = {}
        for r in ctx.requirements:
            key = re.sub(r"[\s_]+", "", r["id"].upper())
            if key in seen:
                result.add("duplicate_id", "blocker", f"identifier also used by another requirement",
                           r["id"], hint="give every requirement a unique, stable id")
            seen[key] = r["id"]
        reqs = [r for r in ctx.requirements if r["text"]]
        if len(reqs) > 3000:
            result.data["note"] = "pairwise checks limited to the first 3000 requirements"
            reqs = reqs[:3000]
        terms = {r["id"]: {stem(w) for w in content_terms(r["text"])} for r in reqs}
        n_dup = n_conf = 0
        for i, a in enumerate(reqs):
            ta = terms[a["id"]]
            for b in reqs[i + 1:]:
                tb = terms[b["id"]]
                if not ta or not tb:
                    continue
                jac = len(ta & tb) / len(ta | tb)
                if jac < 0.35:
                    continue
                sim = _lexical_similarity(a["text"], b["text"])
                neg_a, neg_b = bool(_NEG.search(a["text"])), bool(_NEG.search(b["text"]))
                if neg_a != neg_b and jac >= 0.5:
                    n_conf += 1
                    result.add("polarity_conflict", "major",
                               f"conflicts with {b['id']}: same statement, opposite polarity",
                               a["id"], evidence={"other": b["id"], "similarity": round(sim, 2)},
                               hint="decide which obligation holds, or state the condition separating them")
                    continue
                qa = {u.lower(): float(v) for v, u in _QTY.findall(a["text"])}
                qb = {u.lower(): float(v) for v, u in _QTY.findall(b["text"])}
                clash = [u for u in set(qa) & set(qb) if qa[u] != qb[u]]
                if clash and jac >= 0.6:
                    n_conf += 1
                    u = clash[0]
                    result.add("value_conflict", "major",
                               f"states {qa[u]:g} {u} where {b['id']} states {qb[u]:g} {u} for "
                               f"what reads as the same quantity", a["id"],
                               evidence={"other": b["id"], "unit": u},
                               hint="confirm the two values apply to different conditions",
                               status="candidate")
                elif sim >= self.DUP:
                    n_dup += 1
                    result.add("duplicate", "major", f"near-duplicate of {b['id']}", a["id"],
                               evidence={"other": b["id"], "similarity": round(sim, 2)},
                               hint="merge, or make the difference explicit")
                elif sim >= self.SIMILAR:
                    result.add("similar", "info", f"overlaps strongly with {b['id']}", a["id"],
                               evidence={"other": b["id"], "similarity": round(sim, 2)})
        result.metrics = {"duplicates": n_dup, "conflicts": n_conf}

    def llm_payload(self, ctx, result):
        return [{"id": r["id"], "text": r["text"]} for r in ctx.requirements if r["text"]]
