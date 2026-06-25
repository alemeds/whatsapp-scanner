#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dictionary loading, parsing, and merging."""

import csv
import io
import streamlit as st
from pathlib import Path

CATEGORY_MAP = {
    'palabras_alta': 'high_risk',
    'palabras_media': 'medium_risk',
    'frases_contexto': 'context_phrases',
    'contexto_laboral': 'work_context',
    'contexto_relacion': 'work_context',
    'contexto_financiero': 'work_context',
    'contexto_agresion': 'work_context',
    'contexto_emocional': 'work_context',
    'contexto_digital': 'work_context',
    'contexto_sustancias': 'work_context'
}

PREDEFINED_DICTIONARY_FILES = {
    "Sexual Harassment": "data/sexual_harassment.csv",
    "Cyberbullying": "data/cyberbullying.csv",
    "Threats and Violence": "data/threats_violence.csv",
    "Drugs": "data/drugs.csv",
    "Infidelity": "data/infidelity.csv",
    "Profanity (Argentina)": "data/profanity_arg.csv",
    "Theft and Fraud": "data/theft_fraud.csv",
    "Suicide and Self-Harm": "data/suicide.csv",
    "Complete (All Categories)": "data/complete.csv",
}


def parse_dictionary_rows(reader):
    """Convert (term, category) rows into risk-classified dictionary."""
    dictionary = {
        'high_risk': [],
        'medium_risk': [],
        'context_phrases': [],
        'work_context': []
    }

    for row in reader:
        if not row or not row[0] or row[0].startswith('#'):
            continue

        if len(row) >= 2:
            term = row[0].strip().lower()
            category = row[1].strip().lower()
            mapped_category = CATEGORY_MAP.get(category, 'medium_risk')
            if term:
                dictionary[mapped_category].append(term)

    return dictionary


def parse_quick_terms(raw_text):
    """Parse comma-separated or newline-separated terms for quick search."""
    terms = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        for row in csv.reader([line], skipinitialspace=True):
            for field in row:
                term = field.strip().lower()
                if term:
                    terms.append(term)
    return terms


def load_dictionary_from_file(uploaded_file):
    """Load dictionary from user-uploaded file."""
    try:
        content = uploaded_file.read().decode('utf-8-sig')

        if uploaded_file.name.endswith('.csv'):
            reader = csv.reader(io.StringIO(content))
        else:
            lines = content.strip().split('\n')
            reader = [line.split(',') for line in lines if ',' in line]

        return parse_dictionary_rows(reader)

    except Exception as e:
        st.error(f"Error loading dictionary: {e}")
        return None


@st.cache_data
def load_predefined_dictionary(filename):
    """Load predefined dictionary from CSV file bundled with the app."""
    path = Path(__file__).parent.parent / filename
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        return parse_dictionary_rows(reader)


def merge_dictionaries(dictionaries):
    """Merge multiple dictionaries into one, removing duplicates."""
    merged = {
        'high_risk': [],
        'medium_risk': [],
        'context_phrases': [],
        'work_context': []
    }

    for dictionary in dictionaries:
        for key in merged:
            for term in dictionary[key]:
                if term not in merged[key]:
                    merged[key].append(term)

    return merged
