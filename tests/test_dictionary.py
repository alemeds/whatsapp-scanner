#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for dictionary loading and merging."""

import pytest
from src.dictionary import parse_dictionary_rows, merge_dictionaries, parse_quick_terms, CATEGORY_MAP


class TestDictionaryParsing:
    """Dictionary row parsing tests."""

    def test_should_map_category_correctly(self):
        """Map Spanish categories to internal risk levels."""
        rows = [
            ['danger', 'palabras_alta'],
            ['bad', 'palabras_media'],
        ]

        result = parse_dictionary_rows(rows)
        assert 'danger' in result['high_risk']
        assert 'bad' in result['medium_risk']

    def test_should_skip_comments(self):
        """Ignore commented lines in dictionary."""
        rows = [
            ['# This is a comment'],
            ['term1', 'palabras_alta'],
        ]

        result = parse_dictionary_rows(rows)
        assert len(result['high_risk']) == 1
        assert 'term1' in result['high_risk']

    def test_should_skip_empty_lines(self):
        """Ignore empty rows."""
        rows = [
            [''],
            ['term1', 'palabras_alta'],
            [],
        ]

        result = parse_dictionary_rows(rows)
        assert len(result['high_risk']) == 1


class TestDictionaryMerging:
    """Dictionary merge tests."""

    def test_should_merge_without_duplicates(self):
        """Combine multiple dictionaries, removing duplicates."""
        dict1 = {
            'high_risk': ['danger', 'threat'],
            'medium_risk': [],
            'context_phrases': [],
            'work_context': []
        }

        dict2 = {
            'high_risk': ['danger', 'attack'],
            'medium_risk': ['bad'],
            'context_phrases': [],
            'work_context': []
        }

        merged = merge_dictionaries([dict1, dict2])

        assert len(merged['high_risk']) == 3
        assert 'danger' in merged['high_risk']
        assert 'threat' in merged['high_risk']
        assert 'attack' in merged['high_risk']
        assert len(merged['medium_risk']) == 1


class TestQuickTerms:
    """Quick term parsing tests."""

    def test_should_parse_comma_separated_terms(self):
        """Parse comma-separated terms."""
        text = "juan, pedro, marla"
        terms = parse_quick_terms(text)
        assert len(terms) == 3
        assert 'juan' in terms
        assert 'pedro' in terms

    def test_should_parse_newline_separated_terms(self):
        """Parse newline-separated terms."""
        text = "juan\npedro\nmarla"
        terms = parse_quick_terms(text)
        assert len(terms) == 3

    def test_should_ignore_empty_lines(self):
        """Skip empty entries."""
        text = "juan\n\npedro"
        terms = parse_quick_terms(text)
        assert len(terms) == 2
