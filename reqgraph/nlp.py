"""
reqgraph.nlp
============

PyTorch + transformers intelligence (all imports are lazy, so the package works
without torch installed as long as you don't touch this module).

BertTokenTagger
    A *fine-tunable* BERT token classifier (BIO over requirement roles). You
    label a handful of your own requirements, call ``train``, and it learns to
    tag CONDITION / SUBJECT / MODALITY / ACTOR / PROCESS / OBJECT / CONSTRAINT.
    Plugs into the engine via reqgraph.extractors.BertTaggerExtractor; the
    lossless tiler still guarantees byte-exact round-trip.

RequirementAnalyzer
    BERT sentence embeddings for requirement-level intelligence: semantic
    similarity, near-duplicate detection, and consistency/conflict screening
    (high similarity + opposite polarity) -- useful for DO-178C / ARP4754A
    review activities.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# keep the HuggingFace stack quiet (set before transformers is imported)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .core import Role


def _quiet_transformers():
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

# roles the tagger learns (DETAILS is left to the rule/spacy backends)
TAG_ROLES = [Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
             Role.PROCESS, Role.OBJECT, Role.CONSTRAINT]


def _build_label_maps():
    labels = ["O"]
    for r in TAG_ROLES:
        labels += [f"B-{r.value}", f"I-{r.value}"]
    return labels, {l: i for i, l in enumerate(labels)}, {i: l for i, l in enumerate(labels)}


LABELS, LABEL2ID, ID2LABEL = _build_label_maps()
ROLE_BY_NAME = {r.value: r for r in TAG_ROLES}

# Span = (start, end, Role)


class BertTokenTagger:
    """Fine-tunable BERT token classifier for requirement elements."""

    def __init__(self, model_name: str = "prajjwal1/bert-tiny",
                 tokenizer_name: Optional[str] = None,
                 device: Optional[str] = None, max_len: int = 64):
        self.model_name = model_name
        # tiny BERT checkpoints often omit a fast-tokenizer file; the standard
        # bert-base-uncased fast tokenizer (same WordPiece vocab) supplies the
        # offset-mapping we need and is only a ~0.5 MB download.
        self.tokenizer_name = tokenizer_name or "bert-base-uncased"
        self.max_len = max_len
        self._device = device
        self._tok = None
        self._model = None
        self._torch = None

    # -- lazy model construction ---------------------------------------------
    def _ensure(self):
        if self._model is not None:
            return
        _quiet_transformers()
        import torch
        from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                                  BertConfig)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # BertConfig explicitly: some tiny checkpoints omit "model_type",
        # which breaks AutoConfig's auto-detection.
        config = BertConfig.from_pretrained(
            self.model_name, num_labels=len(LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID)
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, config=config, ignore_mismatched_sizes=True)
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    # -- char spans -> per-token BIO labels (offset aligned) -----------------
    def _encode(self, text, spans):
        enc = self._tok(text, return_offsets_mapping=True, truncation=True,
                        max_length=self.max_len, padding="max_length")
        labels = []
        prev_role = None
        for (cs, ce) in enc["offset_mapping"]:
            if cs == ce:                       # special token / padding
                labels.append(-100)
                prev_role = None
                continue
            role = next((r for (s, e, r) in spans if s <= cs < e), None)
            if role is None:
                labels.append(LABEL2ID["O"])
                prev_role = None
            else:
                prefix = "I-" if prev_role is role else "B-"
                labels.append(LABEL2ID[prefix + role.value])
                prev_role = role
        return enc["input_ids"], enc["attention_mask"], labels

    # -- training -------------------------------------------------------------
    @staticmethod
    def _validate_examples(examples):
        if not examples:
            raise ValueError("training requires at least one labelled example")
        for i, (text, spans) in enumerate(examples):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"example {i}: text must be a non-empty string")
            for (s, e, role) in spans:
                if not (0 <= s < e <= len(text)):
                    raise ValueError(
                        f"example {i}: span ({s},{e}) outside text of length {len(text)}")
                if not isinstance(role, Role):
                    raise ValueError(f"example {i}: span role must be a Role, got {role!r}")

    def train(self, examples, epochs: int = 15, lr: float = 5e-4,
              batch_size: int = 8, verbose: bool = True, seed: Optional[int] = 42):
        """Fine-tune on ``[(text, [(start, end, Role), ...]), ...]``.

        ``seed`` fixes torch's RNG for reproducible runs (None = nondeterministic).
        """
        self._validate_examples(examples)
        self._ensure()
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        from torch.utils.data import DataLoader, TensorDataset

        ids, masks, labs = [], [], []
        for text, spans in examples:
            a, b, c = self._encode(text, spans)
            ids.append(a); masks.append(b); labs.append(c)
        ds = TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(labs))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr)

        self._model.train()
        for ep in range(epochs):
            total = 0.0
            for bid, bam, bll in dl:
                bid, bam, bll = bid.to(self._device), bam.to(self._device), bll.to(self._device)
                out = self._model(input_ids=bid, attention_mask=bam, labels=bll)
                out.loss.backward()
                opt.step()
                opt.zero_grad()
                total += out.loss.item()
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"    epoch {ep + 1:3d}/{epochs}  loss={total / len(dl):.4f}")
        self._model.eval()
        return self

    # -- inference: text -> char spans ---------------------------------------
    def predict_spans(self, text):
        self._ensure()
        torch = self._torch
        enc = self._tok(text, return_offsets_mapping=True, truncation=True,
                        max_length=self.max_len, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        if enc["input_ids"].shape[1] >= self.max_len:
            logger.warning(
                "text exceeds max_len=%d tokens; the tail is left untagged "
                "(round-trip stays lossless, tail becomes GLUE)", self.max_len)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            pred = self._model(**enc).logits[0].argmax(-1).tolist()

        spans, cur = [], None
        for (cs, ce), pid in zip(offsets, pred):
            if cs == ce:
                continue
            lab = ID2LABEL[pid]
            if lab == "O":
                cur = None
                continue
            prefix, rname = lab.split("-", 1)
            role = ROLE_BY_NAME[rname]
            if prefix == "B" or cur is None or cur[2] is not role:
                cur = [cs, ce, role]
                spans.append(cur)
            else:
                cur[1] = ce
        return [(s, e, r) for (s, e, r) in spans]

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, examples) -> dict:
        """Exact-span precision/recall/F1 against labelled examples.

        Use this as a quality gate before deploying a trained tagger
        (e.g. require f1 >= 0.9 on a held-out set)."""
        self._validate_examples(examples)
        tp = fp = fn = 0
        for text, gold in examples:
            pred = set(self.predict_spans(text))
            gold_set = {(s, e, r) for (s, e, r) in gold}
            tp += len(pred & gold_set)
            fp += len(pred - gold_set)
            fn += len(gold_set - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "gold_spans": tp + fn}

    # -- persistence ----------------------------------------------------------
    def save(self, path: str):
        self._ensure()
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        self._tok.save_pretrained(path)
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        obj = cls(model_name=path, tokenizer_name=path, device=device)
        obj._ensure()
        return obj


class RequirementAnalyzer:
    """BERT-embedding requirement-level intelligence."""

    def __init__(self, model_name: str = "prajjwal1/bert-tiny",
                 tokenizer_name: Optional[str] = None,
                 device: Optional[str] = None, max_len: int = 64):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or "bert-base-uncased"
        self.max_len = max_len
        self._device = device
        self._tok = None
        self._model = None
        self._torch = None

    def _ensure(self):
        if self._model is not None:
            return
        _quiet_transformers()
        import torch
        from transformers import AutoModel, AutoTokenizer, BertConfig
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        config = BertConfig.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name, config=config)
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def embed(self, texts):
        """Mean-pooled, L2-normalised sentence embeddings (numpy array)."""
        import numpy as np
        self._ensure()
        torch = self._torch
        if isinstance(texts, str):
            texts = [texts]
        enc = self._tok(list(texts), padding=True, truncation=True,
                        max_length=self.max_len, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def similarity(self, a: str, b: str) -> float:
        e = self.embed([a, b])
        return float(e[0] @ e[1])

    def find_duplicates(self, reqs, threshold: float = 0.9):
        """Return (i, j, score) pairs whose cosine similarity >= threshold."""
        reqs = list(reqs)
        e = self.embed(reqs)
        sim = e @ e.T
        out = [(i, j, float(sim[i, j]))
               for i in range(len(reqs)) for j in range(i + 1, len(reqs))
               if sim[i, j] >= threshold]
        return sorted(out, key=lambda x: -x[2])

    def detect_conflicts(self, reqs, threshold: float = 0.8):
        """Flag highly-similar requirements with opposite polarity (negation)."""
        import re
        reqs = list(reqs)

        def negated(t):
            return bool(re.search(r"\b(shall not|should not|must not|may not|not|never|no)\b",
                                  t, re.I))

        e = self.embed(reqs)
        sim = e @ e.T
        out = [(i, j, float(sim[i, j]))
               for i in range(len(reqs)) for j in range(i + 1, len(reqs))
               if sim[i, j] >= threshold and negated(reqs[i]) != negated(reqs[j])]
        return sorted(out, key=lambda x: -x[2])
