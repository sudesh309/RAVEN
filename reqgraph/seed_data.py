"""
reqgraph.seed_data
==================

A small, hand-labelled aerospace requirement corpus for the BERT token-tagger,
so it is usable out of the box. Each example is written as

    (text, [(Role, substring), ...])

and converted to character spans by index lookup. ``validate()`` asserts every
labelled substring is unique within its sentence (so the spans are unambiguous).

Extend this list with your own labelled requirements, or load your own data and
pass it straight to ``BertTokenTagger.train`` -- the format is
``[(text, [(start, end, Role), ...]), ...]``.
"""

from __future__ import annotations

from .core import Role

# (text, [(Role, exact substring), ...])
_RAW = [
    ("The navigation system shall compute the aircraft position.",
     [(Role.SUBJECT, "The navigation system"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "compute"), (Role.OBJECT, "the aircraft position")]),
    ("The autopilot shall maintain the selected heading.",
     [(Role.SUBJECT, "The autopilot"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "maintain"), (Role.OBJECT, "the selected heading")]),
    ("The flight management system shall calculate the optimal cruise altitude.",
     [(Role.SUBJECT, "The flight management system"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "calculate"), (Role.OBJECT, "the optimal cruise altitude")]),
    ("The display unit shall present the current airspeed.",
     [(Role.SUBJECT, "The display unit"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "present"), (Role.OBJECT, "the current airspeed")]),
    ("The sensor shall measure the cabin pressure.",
     [(Role.SUBJECT, "The sensor"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "measure"), (Role.OBJECT, "the cabin pressure")]),
    ("The radio shall transmit the position report.",
     [(Role.SUBJECT, "The radio"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "transmit"), (Role.OBJECT, "the position report")]),
    ("The recorder shall store the flight data.",
     [(Role.SUBJECT, "The recorder"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "store"), (Role.OBJECT, "the flight data")]),
    ("The actuator shall extend the landing gear.",
     [(Role.SUBJECT, "The actuator"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "extend"), (Role.OBJECT, "the landing gear")]),
    ("The controller shall regulate the fuel flow.",
     [(Role.SUBJECT, "The controller"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "regulate"), (Role.OBJECT, "the fuel flow")]),
    ("The monitor shall detect the system fault.",
     [(Role.SUBJECT, "The monitor"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "detect"), (Role.OBJECT, "the system fault")]),
    ("When the altitude exceeds 10,000 feet, the system shall disable the alert within 2 seconds.",
     [(Role.CONDITION, "When the altitude exceeds 10,000 feet"), (Role.SUBJECT, "the system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "disable"), (Role.OBJECT, "the alert"),
      (Role.CONSTRAINT, "within 2 seconds")]),
    ("When a fault is detected, the controller shall isolate the faulty unit within 100 milliseconds.",
     [(Role.CONDITION, "When a fault is detected"), (Role.SUBJECT, "the controller"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "isolate"), (Role.OBJECT, "the faulty unit"),
      (Role.CONSTRAINT, "within 100 milliseconds")]),
    ("When the engine fire is detected, the engine control unit shall shut off the affected engine within 500 milliseconds.",
     [(Role.CONDITION, "When the engine fire is detected"), (Role.SUBJECT, "the engine control unit"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "shut off"), (Role.OBJECT, "the affected engine"),
      (Role.CONSTRAINT, "within 500 milliseconds")]),
    ("When the cabin altitude exceeds 14,000 feet, the oxygen system shall deploy the oxygen masks within 4 seconds.",
     [(Role.CONDITION, "When the cabin altitude exceeds 14,000 feet"), (Role.SUBJECT, "the oxygen system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "deploy"), (Role.OBJECT, "the oxygen masks"),
      (Role.CONSTRAINT, "within 4 seconds")]),
    ("When the gear is selected down, the actuator shall extend the main landing gear within 10 seconds.",
     [(Role.CONDITION, "When the gear is selected down"), (Role.SUBJECT, "the actuator"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "extend"), (Role.OBJECT, "the main landing gear"),
      (Role.CONSTRAINT, "within 10 seconds")]),
    ("When the temperature exceeds the limit, the cooling system shall activate the ventilation fan.",
     [(Role.CONDITION, "When the temperature exceeds the limit"), (Role.SUBJECT, "the cooling system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "activate"), (Role.OBJECT, "the ventilation fan")]),
    ("While the aircraft is airborne, the system shall monitor the engine speed.",
     [(Role.CONDITION, "While the aircraft is airborne"), (Role.SUBJECT, "the system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "monitor"), (Role.OBJECT, "the engine speed")]),
    ("While the autopilot is engaged, the controller shall hold the target altitude.",
     [(Role.CONDITION, "While the autopilot is engaged"), (Role.SUBJECT, "the controller"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "hold"), (Role.OBJECT, "the target altitude")]),
    ("While taxiing, the system shall limit the ground speed.",
     [(Role.CONDITION, "While taxiing"), (Role.SUBJECT, "the system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "limit"), (Role.OBJECT, "the ground speed")]),
    ("If the signal is lost, the receiver shall trigger the backup channel.",
     [(Role.CONDITION, "If the signal is lost"), (Role.SUBJECT, "the receiver"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "trigger"), (Role.OBJECT, "the backup channel")]),
    ("If the pressure drops below the threshold, the valve shall close the supply line within 5 seconds.",
     [(Role.CONDITION, "If the pressure drops below the threshold"), (Role.SUBJECT, "the valve"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "close"), (Role.OBJECT, "the supply line"),
      (Role.CONSTRAINT, "within 5 seconds")]),
    ("If an overspeed is detected, the system shall reduce the engine thrust.",
     [(Role.CONDITION, "If an overspeed is detected"), (Role.SUBJECT, "the system"),
      (Role.MODALITY, "shall"), (Role.PROCESS, "reduce"), (Role.OBJECT, "the engine thrust")]),
    ("The braking system shall reduce the aircraft speed within 3 seconds.",
     [(Role.SUBJECT, "The braking system"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "reduce"), (Role.OBJECT, "the aircraft speed"),
      (Role.CONSTRAINT, "within 3 seconds")]),
    ("The data bus shall transmit the telemetry packet at a rate of 50 hertz.",
     [(Role.SUBJECT, "The data bus"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "transmit"), (Role.OBJECT, "the telemetry packet"),
      (Role.CONSTRAINT, "at a rate of 50 hertz")]),
    ("The processor shall complete the computation within 20 milliseconds.",
     [(Role.SUBJECT, "The processor"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "complete"), (Role.OBJECT, "the computation"),
      (Role.CONSTRAINT, "within 20 milliseconds")]),
    ("The ground station shall acknowledge the telemetry packet within 2 seconds.",
     [(Role.SUBJECT, "The ground station"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "acknowledge"), (Role.OBJECT, "the telemetry packet"),
      (Role.CONSTRAINT, "within 2 seconds")]),
    ("The warning system shall illuminate the master caution light.",
     [(Role.SUBJECT, "The warning system"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "illuminate"), (Role.OBJECT, "the master caution light")]),
    ("The fuel system shall maintain the tank balance.",
     [(Role.SUBJECT, "The fuel system"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "maintain"), (Role.OBJECT, "the tank balance")]),
    ("The transponder shall report the aircraft identity.",
     [(Role.SUBJECT, "The transponder"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "report"), (Role.OBJECT, "the aircraft identity")]),
    ("The power supply shall provide the backup voltage.",
     [(Role.SUBJECT, "The power supply"), (Role.MODALITY, "shall"),
      (Role.PROCESS, "provide"), (Role.OBJECT, "the backup voltage")]),
]


def _to_spans(text, pairs):
    spans = []
    for role, sub in pairs:
        if text.count(sub) != 1:
            raise ValueError(f"ambiguous/missing label {sub!r} in {text!r}")
        i = text.index(sub)
        spans.append((i, i + len(sub), role))
    return text, spans


def training_examples():
    """Return [(text, [(start, end, Role), ...]), ...] ready for BertTokenTagger.train."""
    return [_to_spans(text, pairs) for text, pairs in _RAW]


def validate():
    """Raise if any labelled substring is ambiguous; return example count."""
    data = training_examples()
    return len(data)


if __name__ == "__main__":
    print(f"seed corpus OK: {validate()} labelled requirements")
