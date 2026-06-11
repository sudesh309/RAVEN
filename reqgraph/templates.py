"""
reqgraph.templates
===================

A ``Template`` is a configurable *requirement structure* (IREB/Rupp, EARS, or
your own). It supplies the keyword/marker sets the extractors anchor on and the
slot order the renderer lays out. Defining a custom structure = one dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .core import Role


@dataclass(frozen=True)
class Template:
    """Immutable (hashable) so compiled regex tables can be cached per template.

    To tweak a shipped template, derive a new one:
        dataclasses.replace(RUPP_TEMPLATE, name="Mine", conjunctions=("and",))
    """
    name: str
    condition_markers: tuple = (
        "when", "while", "if", "as soon as", "once", "after", "before",
        "during", "in case of", "unless", "as long as", "provided that",
        "given that", "whenever",
    )
    modality_keywords: tuple = (
        "shall not", "should not", "must not", "will not", "may not",
        "shall", "should", "must", "will", "may", "cannot", "can",
    )
    # Rupp type 2 (user interaction): "provide <whom> with the ability to <verb>"
    user_interaction_open: tuple = ("provide",)
    user_interaction_bridge: tuple = ("with the ability to",)
    # Rupp type 3 (interface): "be able to <verb>"
    interface_markers: tuple = ("be able to",)
    constraint_markers: tuple = (
        "within", "in less than", "in no more than", "no later than",
        "in at most", "at a rate of", "with an accuracy of",
        "with a resolution of", "with a precision of", "for at least",
        "for no longer than", "every",
    )
    object_determiners: tuple = (
        "the", "a", "an", "its", "their", "each", "every", "all", "any",
        "this", "that", "these", "those", "his", "her", "our", "your",
    )
    conjunctions: tuple = ("and", "or")
    slot_order: tuple = (
        Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.ACTOR,
        Role.PROCESS, Role.OBJECT, Role.DETAILS, Role.CONSTRAINT,
    )

    def alt(self, items) -> str:
        """Regex alternation of markers, longest first (so multi-word wins)."""
        return "|".join(re.escape(w) for w in sorted(items, key=len, reverse=True))


# IREB / Rupp "MASTeR" boilerplate.
RUPP_TEMPLATE = Template(name="IREB-Rupp")

# EARS - Easy Approach to Requirements Syntax (common in aerospace).
EARS_TEMPLATE = Template(
    name="EARS",
    condition_markers=("when", "while", "if", "where", "as soon as"),
    user_interaction_open=(),
    interface_markers=(),
    slot_order=(Role.CONDITION, Role.SUBJECT, Role.MODALITY, Role.PROCESS,
                Role.OBJECT, Role.CONSTRAINT),
)

TEMPLATES: dict[str, Template] = {t.name: t for t in (RUPP_TEMPLATE, EARS_TEMPLATE)}


def register_template(template: Template) -> None:
    """Register a custom requirement structure so it can be used by name."""
    TEMPLATES[template.name] = template
