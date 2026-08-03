"""Adapters for the three dependency parsers used by SUMoE."""

from .base import DependencyParser, ParsedSentence
from .spacy_parser import SpacyParserAdapter
from .stanza_parser import StanzaParserAdapter
from .transition_parser import TransitionParserAdapter

__all__ = [
    "DependencyParser",
    "ParsedSentence",
    "SpacyParserAdapter",
    "StanzaParserAdapter",
    "TransitionParserAdapter",
]

