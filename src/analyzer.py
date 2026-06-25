#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Message risk analysis and scoring."""

import re
from functools import lru_cache

SATURATION_CAP = 3


def setup_sensitivity(level, custom_threshold=None):
    """Configure sensitivity levels."""
    configs = {
        'low': {
            'threshold': 0.75, 'high_weight': 0.6, 'medium_weight': 0.2,
            'context_weight': 0.4, 'work_weight': 0.3, 'bonus': 0.1
        },
        'medium': {
            'threshold': 0.60, 'high_weight': 0.7, 'medium_weight': 0.3,
            'context_weight': 0.5, 'work_weight': 0.4, 'bonus': 0.15
        },
        'high': {
            'threshold': 0.45, 'high_weight': 0.8, 'medium_weight': 0.4,
            'context_weight': 0.6, 'work_weight': 0.5, 'bonus': 0.2
        }
    }

    config = configs.get(level, configs['medium'])
    if custom_threshold is not None:
        config['threshold'] = custom_threshold
    return config


def saturating_ratio(matches, cap=SATURATION_CAP):
    """Convert match count to ratio [0, 1], saturating at cap.
    Ensures small and large dictionaries have comparable scoring.
    """
    return min(matches, cap) / cap


@lru_cache(maxsize=4096)
def _term_pattern(term):
    """Compile and cache regex for matching term as complete word."""
    return re.compile(r'\b' + re.escape(term) + r'\b')


def term_matches(term, text_lower):
    """Check if term appears as complete word in text (not substring match)."""
    return _term_pattern(term).search(text_lower) is not None


def analyze_message(text, sender, config, dictionary):
    """Analyze single message and return risk score, label, and detected terms."""
    text_lower = text.lower()

    high_matches = sum(1 for term in dictionary['high_risk'] if term_matches(term, text_lower))
    medium_matches = sum(1 for term in dictionary['medium_risk'] if term_matches(term, text_lower))
    context_matches = sum(1 for phrase in dictionary['context_phrases'] if term_matches(phrase, text_lower))
    work_matches = sum(1 for term in dictionary['work_context'] if term_matches(term, text_lower))

    high_score = saturating_ratio(high_matches) * config['high_weight']
    medium_score = saturating_ratio(medium_matches) * config['medium_weight']
    context_score = saturating_ratio(context_matches) * config['context_weight']
    work_score = saturating_ratio(work_matches) * config['work_weight']

    bonus = 0
    if high_matches > 0 and context_matches > 0:
        bonus += config['bonus'] * 1.5
    if high_matches > 0 and work_matches > 0:
        bonus += config['bonus'] * 2.0
    if medium_matches > 0 and context_matches > 0:
        bonus += config['bonus'] * 1.0

    risk_score = min(high_score + medium_score + context_score + work_score + bonus, 1.0)

    detected_words = []
    for category, terms in dictionary.items():
        for term in terms:
            if term_matches(term, text_lower) and term not in detected_words:
                detected_words.append(term)

    label = "DETECTED" if risk_score > config['threshold'] else "NOT DETECTED"
    return risk_score, label, detected_words
