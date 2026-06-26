#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Internationalization (i18n) utilities for multi-language support."""

import json
import streamlit as st
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "locales"
SUPPORTED_LANGUAGES = {
    "Español": "es",
    "English": "en"
}

FALLBACK_TRANSLATIONS = {}


def load_translations(language_code):
    """Load translation dictionary for the specified language."""
    locale_file = LOCALES_DIR / f"{language_code}.json"

    if not locale_file.exists():
        if language_code == "es":
            return FALLBACK_TRANSLATIONS
        return load_translations("es")

    try:
        with open(locale_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        if language_code == "es":
            return FALLBACK_TRANSLATIONS
        return load_translations("es")


def get_language():
    """Get current language from session state, default to Spanish."""
    if "language" not in st.session_state:
        st.session_state.language = "es"
    return st.session_state.language


def set_language(language_code):
    """Set the current language in session state."""
    st.session_state.language = language_code


def t(key_path, **kwargs):
    """
    Translate a key from the current language's translations.

    Args:
        key_path: Dot-separated path to the translation key (e.g., "app.title")
        **kwargs: Format variables to substitute in the translation string

    Returns:
        Translated string with variables substituted
    """
    language = get_language()
    translations = load_translations(language)

    keys = key_path.split('.')
    value = translations

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return f"[Missing translation: {key_path}]"

    if value is None:
        return f"[Missing translation: {key_path}]"

    if kwargs:
        return value.format(**kwargs)
    return value
