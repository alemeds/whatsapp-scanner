#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for message risk analysis."""

import pytest
from src.analyzer import analyze_message, setup_sensitivity, saturating_ratio, term_matches


class TestSensitivity:
    """Sensitivity configuration tests."""

    def test_should_setup_low_sensitivity(self):
        """Configure low sensitivity level."""
        config = setup_sensitivity('low')
        assert config['threshold'] == 0.75

    def test_should_setup_medium_sensitivity(self):
        """Configure medium sensitivity level."""
        config = setup_sensitivity('medium')
        assert config['threshold'] == 0.60

    def test_should_setup_high_sensitivity(self):
        """Configure high sensitivity level."""
        config = setup_sensitivity('high')
        assert config['threshold'] == 0.45

    def test_should_override_with_custom_threshold(self):
        """Use custom threshold when provided."""
        config = setup_sensitivity('low', custom_threshold=0.5)
        assert config['threshold'] == 0.5


class TestSaturation:
    """Saturation ratio tests."""

    def test_should_saturate_at_cap(self):
        """Ratio caps at saturation point."""
        ratio = saturating_ratio(5)
        assert ratio == 1.0

    def test_should_scale_below_cap(self):
        """Ratio scales linearly below cap."""
        ratio = saturating_ratio(1)
        assert ratio == pytest.approx(1/3)


class TestTermMatching:
    """Term matching tests."""

    def test_should_detect_high_risk_term(self):
        """Identify high-risk terms in message."""
        dictionary = {
            'high_risk': ['danger', 'threat'],
            'medium_risk': [],
            'context_phrases': [],
            'work_context': []
        }
        config = setup_sensitivity('medium')

        risk, label, words = analyze_message('This is danger', 'Alice', config, dictionary)

        assert risk > 0
        assert 'danger' in words
        assert label == "NOT DETECTED"  # below threshold

    def test_should_not_detect_substring_match(self):
        """Avoid false positives from partial word matches."""
        dictionary = {
            'high_risk': ['cama'],
            'medium_risk': [],
            'context_phrases': [],
            'work_context': []
        }
        config = setup_sensitivity('medium')

        risk, label, words = analyze_message('Mi cámara no funciona', 'Alice', config, dictionary)

        assert 'cama' not in words

    def test_should_apply_saturation_cap(self):
        """Use saturation to prevent dictionary size inflation."""
        dict_small = {
            'high_risk': ['bad'],
            'medium_risk': [],
            'context_phrases': [],
            'work_context': []
        }

        dict_large = {
            'high_risk': ['bad'] + [f'filler{i}' for i in range(100)],
            'medium_risk': [],
            'context_phrases': [],
            'work_context': []
        }

        config = setup_sensitivity('medium')

        risk1, _, _ = analyze_message('This is bad', 'Alice', config, dict_small)
        risk2, _, _ = analyze_message('This is bad', 'Alice', config, dict_large)

        assert risk1 == risk2

    def test_should_detect_with_combinations(self):
        """Apply bonus when high-risk + context found."""
        dictionary = {
            'high_risk': ['danger'],
            'medium_risk': [],
            'context_phrases': ['at work'],
            'work_context': []
        }
        config = setup_sensitivity('medium')

        risk1, _, _ = analyze_message('danger', 'Alice', config, dictionary)
        risk2, _, _ = analyze_message('danger at work', 'Alice', config, dictionary)

        assert risk2 > risk1
