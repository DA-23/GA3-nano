"""Trimmed package init for GA3-nano.

Only the three lightweight modules used by the Bayes runners are imported here,
so the package loads without the heavy training stack (torch / sklearn /
transformers / spaCy) that the upstream ``multilingual_nn`` package pulls in.
"""
from .languages import ACTIVE_LANGUAGE_SET, ALL_LANGUAGES, LANGUAGES, LOW_RESOURCE_LANGUAGES, NUM_LANGUAGES, LanguageSpec
from .phrase_data import PhraseSegmenter, normalize_english_text, should_passthrough_phrase
from .phrase_translation import (
    GoogleTranslatePhraseTranslator,
    PhraseTranslation,
    PhraseTranslationResult,
)

__all__ = [
    "ACTIVE_LANGUAGE_SET",
    "ALL_LANGUAGES",
    "LANGUAGES",
    "LOW_RESOURCE_LANGUAGES",
    "NUM_LANGUAGES",
    "LanguageSpec",
    "PhraseSegmenter",
    "normalize_english_text",
    "should_passthrough_phrase",
    "GoogleTranslatePhraseTranslator",
    "PhraseTranslation",
    "PhraseTranslationResult",
]
