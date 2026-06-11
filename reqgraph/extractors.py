"""
reqgraph.extractors
====================

Pluggable extraction backends (Strategy pattern). Every backend implements

    extract(text, template) -> list[(start, end, Role, attrs)]

and only *proposes* typed character spans -- the lossless tiler does the rest,
so swapping backends can never corrupt a requirement.

Backends
--------
RuleExtractor        deterministic anchors (default; zero dependencies)
SpacyExtractor       spaCy dependency parse (handles complex/nested sentences)
BertTaggerExtractor  fine-tuned BERT token classifier (see reqgraph.nlp)

Performance: all regexes for a template are compiled once and memoised in
``_patterns`` (templates are frozen/hashable), so batch parsing pays no
recompilation cost.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from .core import OBLIGATION, Role
from .errors import ExtractionError
from .templates import Template

logger = logging.getLogger(__name__)

_FLAGS = re.IGNORECASE


@lru_cache(maxsize=128)
def _patterns(t: Template) -> dict:
    """Compiled regex table for a template (memoised; Template is frozen)."""

    def marker(items):
        return re.compile(r"(?<![\w'])(" + t.alt(items) + r")(?![\w'])", _FLAGS) \
            if items else None

    p = {
        "cond_lead": (re.compile(r"^(\s*)(" + t.alt(t.condition_markers) + r")(?![\w'])",
                                 _FLAGS) if t.condition_markers else None),
        "cond": marker(t.condition_markers),
        "modality": marker(t.modality_keywords),
        "constraint": marker(t.constraint_markers),
        "determiner": marker(t.object_determiners),
        "interface": marker(t.interface_markers),
        "conjunction": (re.compile(r"\s+(" + t.alt(t.conjunctions) + r")\s+", _FLAGS)
                        if t.conjunctions else None),
        "ui": None,
        "then": re.compile(r"(?<![\w'])then(?![\w'])", _FLAGS),
        "then_lead": re.compile(r"\s*then(?![\w'])", _FLAGS),
        "word": re.compile(r"\s*([\w']+)"),
    }
    if t.user_interaction_open and t.user_interaction_bridge:
        p["ui"] = re.compile(
            r"(?<![\w'])(" + t.alt(t.user_interaction_open) + r")\s+(.+?)\s+(" +
            t.alt(t.user_interaction_bridge) + r")(?![\w'])", _FLAGS)
    return p


class Extractor(ABC):
    """Extraction backend interface. Implementations must be side-effect free
    on the input text and return claims as (start, end, Role, attrs) tuples."""

    name = "base"

    def available(self) -> bool:
        return True

    @abstractmethod
    def extract(self, text: str, template: Template) -> list:
        ...

    @staticmethod
    def _trim(text, s, e):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e


def _user_interaction_actor(text, template, start=0, end=None):
    """Locate the 'whom' in 'provide <whom> with the ability to' (Rupp type 2).

    Returns (actor_start, actor_end, action_start) or None."""
    pat = _patterns(template)["ui"]
    if pat is None:
        return None
    m = pat.search(text, start, len(text) if end is None else end)
    if not m:
        return None
    s, e = Extractor._trim(text, m.start(2), m.end(2))
    return (s, e, m.end(3))


# ---------------------------------------------------------------------------
# 1. Rule-based extractor (default)
# ---------------------------------------------------------------------------

class RuleExtractor(Extractor):
    """Deterministic anchor heuristics following the template's keyword sets.

    Stage order (each stage narrows the remaining window):
      condition -> modality (+subject) -> trailing condition -> constraint
      -> function type (actor) -> action core (process/object, compound split)
    """

    name = "rules"

    def _parse_action(self, text, a_start, a_end, t, function_type):
        p = _patterns(t)
        a_start, a_end = self._trim(text, a_start, a_end)
        claims = []

        # a top-level conjunction followed by a non-determiner word starts a
        # second action ("...shut off the engine AND activate the system")
        split_at = None
        if p["conjunction"] is not None:
            for m in p["conjunction"].finditer(text, a_start, a_end):
                nxt = p["word"].match(text, m.end(), a_end)
                first_word = nxt.group(1).lower() if nxt else ""
                if first_word and first_word not in t.object_determiners:
                    split_at = (m.start(), m.end())
                    break

        def simple(s, e):
            s, e = self._trim(text, s, e)
            d = p["determiner"].search(text, s, e) if p["determiner"] else None
            if d is not None and d.start() > s:
                ps, pe = self._trim(text, s, d.start())
                os_, oe = self._trim(text, d.start(), e)
                if pe > ps:
                    claims.append((ps, pe, Role.PROCESS, {"function_type": function_type}))
                if oe > os_:
                    claims.append((os_, oe, Role.OBJECT, {}))
            elif e > s:
                claims.append((s, e, Role.PROCESS, {"function_type": function_type}))

        if split_at:
            simple(a_start, split_at[0])
            simple(split_at[1], a_end)
        else:
            simple(a_start, a_end)
        return claims

    def extract(self, text, template):
        t, p = template, _patterns(template)
        claims = []
        n = len(text)
        work_start = 0

        # (1) leading condition; locate modality first so the condition can end
        # at the last comma *before* the modality (avoids "14,000" splits).
        mcond = p["cond_lead"].match(text) if p["cond_lead"] else None
        first_comma = text.find(",", mcond.end()) if mcond else -1
        mod_search_start = (first_comma + 1 if first_comma != -1 else mcond.end()) if mcond else 0

        mm = p["modality"].search(text, mod_search_start, n) if p["modality"] else None
        mod_start = mm.start(1) if mm else n

        if mcond:
            last_comma = text.rfind(",", mcond.end(), mod_start)
            stop = last_comma if last_comma != -1 else first_comma
            if stop == -1:  # comma-less conditional, terminated by "then"
                tn = p["then"].search(text, mcond.end(), mod_start)
                stop = tn.start() if tn else -1
            if stop != -1:
                cs, ce = self._trim(text, mcond.start(2), stop)
                claims.append((cs, ce, Role.CONDITION,
                               {"marker": mcond.group(2).lower(), "kind": "pre-condition"}))
                work_start = stop + 1 if text[stop:stop + 1] == "," else stop
                tmt = p["then_lead"].match(text, work_start)
                if tmt:
                    work_start = tmt.end()

        # (2) modality (anchor): text between condition and modality = subject
        post_start = n
        if mm:
            kw = mm.group(1).lower()
            claims.append((mm.start(1), mm.end(1), Role.MODALITY,
                           {"obligation": OBLIGATION.get(kw, "unspecified")}))
            ss, se = self._trim(text, work_start, mm.start(1))
            if se > ss:
                claims.append((ss, se, Role.SUBJECT, {}))
            post_start = mm.end(1)
        else:
            post_start = work_start

        post_end = n
        while post_end > post_start and text[post_end - 1] in ".;\n\t ":
            post_end -= 1

        # (3) trailing condition
        tc = p["cond"].search(text, post_start, post_end) if p["cond"] else None
        if tc and tc.start() > post_start:
            cs, ce = self._trim(text, tc.start(), post_end)
            claims.append((cs, ce, Role.CONDITION,
                           {"marker": tc.group(1).lower(), "kind": "post-condition"}))
            post_end = tc.start()

        # (4) trailing constraint
        cc = p["constraint"].search(text, post_start, post_end) if p["constraint"] else None
        if cc and cc.start() > post_start:
            cs, ce = self._trim(text, cc.start(), post_end)
            claims.append((cs, ce, Role.CONSTRAINT, {"marker": cc.group(1).lower()}))
            post_end = cc.start()

        # (5) function type (Rupp): user-interaction / interface / autonomous
        function_type = "autonomous activity"
        action_start = post_start
        ui = _user_interaction_actor(text, t, post_start, post_end)
        if ui:
            function_type = "user interaction"
            claims.append((ui[0], ui[1], Role.ACTOR, {}))
            action_start = ui[2]
        elif p["interface"] is not None:
            itf = p["interface"].search(text, post_start, post_end)
            if itf:
                function_type = "interface requirement"
                action_start = itf.end(1)

        # (6) action core -> process / object (+ compound split)
        claims.extend(self._parse_action(text, action_start, post_end, t, function_type))
        return claims


# ---------------------------------------------------------------------------
# 2. spaCy dependency-parse extractor
# ---------------------------------------------------------------------------

class SpacyExtractor(Extractor):
    name = "spacy"
    _nlp_cache: dict = {}  # model name -> loaded pipeline (loading is expensive)

    def __init__(self, model: str = "en_core_web_sm"):
        self.model = model

    def available(self) -> bool:
        try:
            import spacy
            return spacy.util.is_package(self.model)
        except Exception:
            return False

    def _load(self):
        nlp = SpacyExtractor._nlp_cache.get(self.model)
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load(self.model)
            except Exception as exc:
                raise ExtractionError(
                    f"spaCy model {self.model!r} could not be loaded; install with "
                    f"'python -m spacy download {self.model}'") from exc
            SpacyExtractor._nlp_cache[self.model] = nlp
            logger.info("loaded spaCy model %s", self.model)
        return nlp

    def extract(self, text, template):
        doc = self._load()(text)
        claims = []
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        if not roots:
            logger.warning("spaCy found no ROOT; falling back to rules")
            return RuleExtractor().extract(text, template)
        root = roots[0]

        def subtree_span(tok):
            toks = list(tok.subtree)
            s = min(x.idx for x in toks)
            e = max(x.idx + len(x) for x in toks)
            return self._trim(text, s, e)

        def clip(s, e, excl):
            for (xs, xe) in excl:
                if s < xs < e:
                    e = xs
            return self._trim(text, s, e)

        # conditions (subordinate adverbial clauses)
        cond_spans = []
        for tok in doc:
            if tok.dep_ == "advcl":
                s, e = subtree_span(tok)
                marker = next((c.text.lower() for c in tok.subtree if c.dep_ == "mark"), "")
                kind = "pre-condition" if s < root.idx else "post-condition"
                claims.append((s, e, Role.CONDITION, {"marker": marker, "kind": kind}))
                cond_spans.append((s, e))

        # constraints (prep phrase headed by a constraint marker)
        constr_spans = []
        for tok in doc:
            if tok.dep_ == "prep" and tok.text.lower() in template.constraint_markers:
                s, e = subtree_span(tok)
                claims.append((s, e, Role.CONSTRAINT, {"marker": tok.text.lower()}))
                constr_spans.append((s, e))

        # actor (Rupp user interaction)
        function_type = "autonomous activity"
        actor_spans = []
        ui = _user_interaction_actor(text, template)
        if ui:
            function_type = "user interaction"
            claims.append((ui[0], ui[1], Role.ACTOR, {}))
            actor_spans.append((ui[0], ui[1]))

        excl = cond_spans + constr_spans + actor_spans

        # modality (modal auxiliary of the root verb)
        for ch in root.children:
            if ch.dep_ in ("aux", "auxpass") and (
                    ch.tag_ == "MD" or ch.text.lower() in template.modality_keywords):
                s, e = ch.idx, ch.idx + len(ch)
                claims.append((s, e, Role.MODALITY,
                               {"obligation": OBLIGATION.get(ch.text.lower(), "unspecified")}))
                break

        # subject
        for ch in root.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                s, e = subtree_span(ch)
                if not any(xs <= s < xe for (xs, xe) in cond_spans):
                    s, e = clip(s, e, excl)
                    if e > s:
                        claims.append((s, e, Role.SUBJECT, {}))
                    break

        # verbs (root + conjoined) -> process + objects
        verbs = [root] + [c for c in root.children if c.dep_ == "conj" and c.pos_ == "VERB"]
        for v in verbs:
            ptoks = [v] + [c for c in v.children if c.dep_ == "prt"]
            ps = min(x.idx for x in ptoks)
            pe = max(x.idx + len(x) for x in ptoks)
            ps, pe = self._trim(text, ps, pe)
            if pe > ps and not any(xs <= ps < xe for (xs, xe) in excl):
                claims.append((ps, pe, Role.PROCESS, {"function_type": function_type}))
            for ch in v.children:
                if ch.dep_ in ("dobj", "obj", "attr", "oprd"):
                    s, e = subtree_span(ch)
                    if not any(xs <= s < xe for (xs, xe) in excl):
                        s, e = clip(s, e, excl)
                        if e > s:
                            claims.append((s, e, Role.OBJECT, {}))
                elif ch.dep_ == "dative":
                    claims.append((*subtree_span(ch), Role.ACTOR, {}))
        return claims


# ---------------------------------------------------------------------------
# 3. BERT token-tagger extractor (wraps reqgraph.nlp.BertTokenTagger)
# ---------------------------------------------------------------------------

class BertTaggerExtractor(Extractor):
    name = "bert"

    def __init__(self, tagger=None, model_dir: str | None = None):
        if tagger is None and model_dir is None:
            raise ExtractionError(
                "BertTaggerExtractor needs a live tagger or a saved model_dir")
        self._tagger = tagger
        self._model_dir = model_dir

    def available(self) -> bool:
        return True

    def _get(self):
        if self._tagger is None:
            from .nlp import BertTokenTagger
            self._tagger = BertTokenTagger.load(self._model_dir)
        return self._tagger

    def extract(self, text, template):
        claims = []
        for (s, e, role) in self._get().predict_spans(text):
            attrs = {}
            if role is Role.MODALITY:
                attrs = {"obligation": OBLIGATION.get(text[s:e].lower(), "unspecified")}
            elif role is Role.PROCESS:
                attrs = {"function_type": "autonomous activity"}
            claims.append((s, e, role, attrs))
        return claims


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

EXTRACTORS = {"rules": RuleExtractor, "spacy": SpacyExtractor, "bert": BertTaggerExtractor}


def get_extractor(name: str, **kwargs) -> Extractor:
    if name not in EXTRACTORS:
        raise ExtractionError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name](**kwargs)


def auto_select() -> Extractor:
    """Best available extractor that needs no training: spaCy, else rules."""
    try:
        ex = SpacyExtractor()
        if ex.available():
            return ex
    except Exception:  # pragma: no cover - defensive
        pass
    return RuleExtractor()
