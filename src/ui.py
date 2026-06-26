#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UI components: alerts, visualizations, and text helpers."""

import streamlit as st
import plotly.express as px
from .i18n import t

AUTHOR_LINKEDIN = "https://www.linkedin.com/in/alemeds/"


def generate_smart_alerts(results_df):
    """Generate alerts based on detection frequency and sender concentration."""
    alerts = []
    detected_df = results_df[results_df['label'] == 'DETECTED']

    if len(detected_df) == 0:
        return alerts

    detection_rate = len(detected_df) / len(results_df)
    if detection_rate > 0.3:
        alerts.append(('error', f"⚠️ CRITICAL: {len(detected_df)} detections of {len(results_df)} messages ({detection_rate:.1%})"))
    elif detection_rate > 0.15:
        alerts.append(('warning', f"🔔 WARNING: {len(detected_df)} detections of {len(results_df)} messages ({detection_rate:.1%})"))

    sender_counts = detected_df['sender'].value_counts()
    if len(sender_counts) > 0:
        dominant_pct = sender_counts.iloc[0] / len(detected_df)
        if dominant_pct > 0.7:
            alerts.append(('info', f"👤 DOMINANT SENDER: {sender_counts.index[0]} accounts for {dominant_pct:.1%} of detections"))

    return alerts


def create_visualizations(results_df, detection_type):
    """Create visualizations of analysis results."""
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            results_df,
            x='risk_score',
            nbins=20,
            title=f'Risk Score Distribution - {detection_type}',
            labels={'risk_score': 'Risk Score', 'count': 'Message Count'}
        )
        fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red",
                          annotation_text="Default Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        detections_by_sender = results_df[results_df['label'] == 'DETECTED']['sender'].value_counts()
        if not detections_by_sender.empty:
            fig_pie = px.pie(
                values=detections_by_sender.values,
                names=detections_by_sender.index,
                title=f'Detections by Sender - {detection_type}'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No detections found to display")


def get_instructions_text():
    """Return user instructions as formatted markdown."""
    return f"""# 🔍 {t('instructions.title')}

## {t('instructions.what_is')}

{t('instructions.what_is_text')}

## {t('instructions.how_to_use')}

### {t('instructions.prepare_chat')}
{t('instructions.prepare_chat_steps')}

### {t('instructions.select_categories')}
{t('instructions.select_categories_text')}

### {t('instructions.adjust_sensitivity')}
{t('instructions.adjust_sensitivity_text')}

### {t('instructions.upload_analyze')}
{t('instructions.upload_analyze_text')}

## {t('instructions.results')}
{t('instructions.results_text')}

## {t('instructions.about_author')}
{t('instructions.author_linkedin', linkedin=AUTHOR_LINKEDIN)}

## {t('instructions.privacy')}
{t('instructions.privacy_text')}
"""
