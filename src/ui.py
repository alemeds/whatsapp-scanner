#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UI components: alerts, visualizations, and text helpers."""

import streamlit as st
import plotly.express as px

SUPPORT_EMAIL = "antoniolmartinez@gmail.com"


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
    return f"""# 🔍 WhatsApp Analyzer — Usage Guide

## What is this tool?

Local detection system for analyzing patterns in WhatsApp conversations.
Processes files entirely without sending them to external servers — everything runs on your machine.

## How to use

### 1. Prepare the chat
- Open WhatsApp and go to the chat you want to analyze
- More options (⋮) → Export chat → Without media
- Save the file (.txt) on your computer

### 2. Select categories or terms
- **Predefined categories:** Sexual Harassment, Cyberbullying, Threats, Drugs, Infidelity, etc.
- **Custom terms:** Type names or words in the "Type" tab
  - Separate with commas or line breaks: `juan, pedro, marla`
  - For exact phrase search, use quotes: `"good morning"`
- **CSV file:** For many terms, upload a CSV with `term,category` per line

### 3. Adjust sensitivity
- **Low:** Fewer false positives, but may miss real cases
- **Medium:** Recommended balance
- **High:** Detects more cases, but may have more false positives

### 4. Upload the chat and analyze
- Upload the .txt file you exported
- The app analyzes automatically and shows results

## Results
- **DETECTED:** Message exceeded risk threshold
- **NOT DETECTED:** Below threshold
- **Detected words:** Which terms were found in each message

## Privacy
✅ All files are processed locally on your machine
✅ Nothing is stored or sent to external servers
✅ Only you see the results

## Support
Questions or reports: {SUPPORT_EMAIL}
"""
