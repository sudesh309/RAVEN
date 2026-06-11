"""
reqgraph_demo.py
================

End-to-end demonstration of the modular reqgraph toolkit:

  1. Rule backend (default)          - lossless round-trip, IREB/Rupp elements
  2. spaCy backend                   - dependency-parse extraction
  3. Trainable BERT token-tagger     - fine-tune (tiny model) + predict
  4. RequirementAnalyzer (BERT)      - similarity / duplicates / conflicts
  5. Quality + type + EARS analysis  - IREB smells & classification
  6. CSV / Excel / ReqIF batch I/O

Run:  python reqgraph_demo.py
"""

from reqgraph import RequirementParser, RUPP_TEMPLATE, EARS_TEMPLATE, Role
from reqgraph.extractors import SpacyExtractor, BertTaggerExtractor

BAR = "=" * 78

AERO = [
    "The flight management system shall calculate the optimal cruise altitude.",
    "When the cabin altitude exceeds 14,000 feet, the oxygen system shall deploy "
    "the passenger oxygen masks within 4 seconds.",
    "While the aircraft is on the ground, the avionics suite shall provide the pilot "
    "with the ability to configure the flight plan if the parking brake is set.",
    "As soon as an engine fire is detected, the engine control unit shall shut off "
    "the affected engine and activate the fire suppression system within 500 milliseconds.",
]


def demo_rules():
    print(BAR, "\n1. RULE BACKEND (default) - lossless round-trip\n", BAR, sep="")
    p = RequirementParser(RUPP_TEMPLATE)
    for t in AERO:
        g = p.parse(t)
        assert g.generate() == t
        print(f"\n{t}")
        print(g.summary())
        print("   round-trip exact :", g.generate() == t)


def demo_spacy():
    print("\n" + BAR, "\n2. SPACY BACKEND - dependency parse\n", BAR, sep="")
    ex = SpacyExtractor()
    if not ex.available():
        print("spaCy model not available - skipped.")
        return
    p = RequirementParser(RUPP_TEMPLATE, ex)
    for t in AERO[:2] + AERO[3:]:
        g = p.parse(t)
        assert g.generate() == t
        print(f"\n{t}")
        print("  ", [(n.role.value, n.text.strip()) for n in g.elements()])
        print("   round-trip exact :", g.generate() == t)


def _labelled_training_data():
    """A few hand-labelled aerospace requirements (char spans) for the tagger.

    Each entry: (text, [(start, end, Role), ...]). Helper below computes spans
    from substrings so it stays readable.
    """
    def spans(text, items):
        out = []
        for sub, role in items:
            i = text.index(sub)
            out.append((i, i + len(sub), role))
        return text, out

    data = [
        spans("The navigation system shall compute the aircraft position.",
              [("The navigation system", Role.SUBJECT), ("shall", Role.MODALITY),
               ("compute", Role.PROCESS), ("the aircraft position", Role.OBJECT)]),
        spans("The autopilot shall maintain the selected heading.",
              [("The autopilot", Role.SUBJECT), ("shall", Role.MODALITY),
               ("maintain", Role.PROCESS), ("the selected heading", Role.OBJECT)]),
        spans("When the altitude exceeds 10,000 feet, the system shall disable the warning within 2 seconds.",
              [("When the altitude exceeds 10,000 feet", Role.CONDITION),
               ("the system", Role.SUBJECT), ("shall", Role.MODALITY),
               ("disable", Role.PROCESS), ("the warning", Role.OBJECT),
               ("within 2 seconds", Role.CONSTRAINT)]),
        spans("When a fault is detected, the controller shall isolate the faulty unit within 100 milliseconds.",
              [("When a fault is detected", Role.CONDITION), ("the controller", Role.SUBJECT),
               ("shall", Role.MODALITY), ("isolate", Role.PROCESS),
               ("the faulty unit", Role.OBJECT), ("within 100 milliseconds", Role.CONSTRAINT)]),
        spans("The display unit shall present the airspeed.",
              [("The display unit", Role.SUBJECT), ("shall", Role.MODALITY),
               ("present", Role.PROCESS), ("the airspeed", Role.OBJECT)]),
        spans("The braking system shall reduce the aircraft speed within 3 seconds.",
              [("The braking system", Role.SUBJECT), ("shall", Role.MODALITY),
               ("reduce", Role.PROCESS), ("the aircraft speed", Role.OBJECT),
               ("within 3 seconds", Role.CONSTRAINT)]),
        spans("When the gear is selected down, the actuator shall extend the landing gear.",
              [("When the gear is selected down", Role.CONDITION), ("the actuator", Role.SUBJECT),
               ("shall", Role.MODALITY), ("extend", Role.PROCESS),
               ("the landing gear", Role.OBJECT)]),
        spans("The sensor shall measure the cabin pressure.",
              [("The sensor", Role.SUBJECT), ("shall", Role.MODALITY),
               ("measure", Role.PROCESS), ("the cabin pressure", Role.OBJECT)]),
    ]
    return data


def demo_bert():
    print("\n" + BAR, "\n3. TRAINABLE BERT TOKEN-TAGGER (PyTorch) - fine-tune + predict\n", BAR, sep="")
    try:
        from reqgraph.nlp import BertTokenTagger
    except Exception as e:
        print("transformers/torch unavailable - skipped:", e)
        return None
    tagger = BertTokenTagger(model_name="prajjwal1/bert-tiny")
    print("Fine-tuning bert-tiny on 8 labelled aerospace requirements...")
    tagger.train(_labelled_training_data(), epochs=40, lr=5e-4, verbose=True)

    held_out = "When the pressure drops, the controller shall close the valve within 5 seconds."
    print(f"\nHeld-out requirement:\n  {held_out}")
    g = RequirementParser(RUPP_TEMPLATE, BertTaggerExtractor(tagger=tagger)).parse(held_out)
    print("  BERT-tagged elements:")
    for n in g.elements():
        print(f"     {n.role.value:<11}: {n.text.strip()!r}")
    print("   round-trip exact :", g.generate() == held_out)
    return tagger


def demo_analyzer():
    print("\n" + BAR, "\n4. REQUIREMENT ANALYZER (BERT embeddings)\n", BAR, sep="")
    try:
        from reqgraph.nlp import RequirementAnalyzer
    except Exception as e:
        print("transformers/torch unavailable - skipped:", e)
        return
    az = RequirementAnalyzer(model_name="prajjwal1/bert-tiny")
    catalogue = [
        "The autopilot shall maintain the selected heading.",
        "The auto-flight system shall hold the chosen heading.",         # ~ duplicate of #0
        "The system shall not deploy the spoilers during takeoff.",
        "The system shall deploy the spoilers during takeoff.",          # conflict with #2
    ]
    print("Catalogue:")
    for i, r in enumerate(catalogue):
        print(f"   [{i}] {r}")
    print("\nNear-duplicates (cosine >= 0.97):")
    for i, j, s in az.find_duplicates(catalogue, threshold=0.97):
        print(f"   [{i}] ~ [{j}]   score={s:.3f}")
    print("Potential conflicts (similar + opposite polarity, >= 0.95):")
    for i, j, s in az.detect_conflicts(catalogue, threshold=0.95):
        print(f"   [{i}] <> [{j}]  score={s:.3f}")


def demo_quality():
    print("\n" + BAR, "\n5. QUALITY / TYPE / EARS ANALYSIS (IREB)\n", BAR, sep="")
    from reqgraph.quality import enrich
    samples = [
        "When an engine fire is detected, the engine control unit shall shut off the engine within 500 milliseconds.",
        "The system should be fast and user-friendly and reliable.",
        "While taxiing, the system shall provide the crew with the ability to select the runway.",
    ]
    p = RequirementParser(RUPP_TEMPLATE)
    for t in samples:
        g = enrich(p.parse(t))
        print(f"\n{t}")
        print(f"   type        : {g.analysis['type']}")
        print(f"   EARS pattern: {g.analysis['ears_pattern']}")
        q = g.analysis["quality"]
        smells = [k for k, v in q.items() if v and k != "weak_words"]
        if q["weak_words"]:
            smells.append(f"weak_words={q['weak_words']}")
        print(f"   smells      : {smells or 'none'}")


def demo_io():
    print("\n" + BAR, "\n6. BATCH I/O - CSV / Excel / ReqIF\n", BAR, sep="")
    import os
    import tempfile
    from reqgraph.io_formats import (write_csv, write_excel, write_reqif,
                                     read_requirements_csv, read_reqif)
    items = [(f"REQ-{i+1:03d}", t) for i, t in enumerate(AERO)]
    d = tempfile.mkdtemp(prefix="reqgraph_")
    csv_p = write_csv(items, os.path.join(d, "reqs.csv"))
    xls_p = write_excel(items, os.path.join(d, "reqs.xlsx"))
    rif_p = write_reqif(items, os.path.join(d, "reqs.reqif"))
    print("   wrote:", csv_p)
    print("   wrote:", xls_p)
    print("   wrote:", rif_p)

    back_csv = read_requirements_csv(csv_p)
    print(f"   CSV reload   : {len(back_csv)} requirements")
    back_rif = read_reqif(rif_p)
    ok = [t for (_id, t) in back_rif] == [t for (_id, t) in items]
    print(f"   ReqIF reload : {len(back_rif)} requirements, text round-trip exact = {ok}")


if __name__ == "__main__":
    demo_rules()
    demo_spacy()
    demo_bert()
    demo_analyzer()
    demo_quality()
    demo_io()
    print("\n" + BAR, "\nAll demos complete.\n", BAR, sep="")
