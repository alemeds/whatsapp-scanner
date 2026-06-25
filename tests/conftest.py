#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_dictionary():
    """Provide a sample dictionary for testing."""
    return {
        'high_risk': ['danger', 'threat'],
        'medium_risk': ['bad'],
        'context_phrases': ['at work'],
        'work_context': ['boss', 'meeting']
    }


@pytest.fixture
def sample_config():
    """Provide a sample sensitivity configuration."""
    from src.analyzer import setup_sensitivity
    return setup_sensitivity('medium')
