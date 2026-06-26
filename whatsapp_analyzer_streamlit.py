#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""WhatsApp Analyzer - Pattern Detection Application."""

import streamlit as st
import pandas as pd
import io
from datetime import datetime

from src.parser import extract_messages_from_text, validate_whatsapp_file, extract_dates_and_senders, parse_whatsapp_date
from src.dictionary import PREDEFINED_DICTIONARY_FILES, load_predefined_dictionary, load_dictionary_from_file, parse_quick_terms, merge_dictionaries
from src.analyzer import analyze_message, setup_sensitivity
from src.ui import generate_smart_alerts, create_visualizations, get_instructions_text

st.set_page_config(
    page_title="WhatsApp Analyzer - Pattern Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 WhatsApp Conversation Analyzer</h1>
        <p>Advanced system for pattern and behavior detection in chats</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Read instructions", expanded=False):
        instructions = get_instructions_text()
        st.markdown(instructions)
        st.download_button(
            label="📥 Download instructions (TXT)",
            data=instructions,
            file_name="whatsapp_analyzer_instructions.txt",
            mime="text/plain"
        )

    with st.sidebar:
        st.header("⚙️ Configuration")

        selected_categories = st.multiselect(
            "🎯 Detection Type(s)",
            options=list(PREDEFINED_DICTIONARY_FILES.keys()),
            help="You can combine multiple categories at once"
        )

        st.subheader("📁 Add Custom Terms (Optional)")

        tab1, tab2 = st.tabs(["📝 Type", "📤 File"])

        extra_dictionary = None

        with tab1:
            quick_terms_text = st.text_area(
                "Type names or words (separated by commas or line breaks)",
                placeholder='juan, pedro\nmarla\n"good morning"',
                height=80,
                help='Each term is searched independently (OR). Quoted text is searched as exact phrase.'
            )
            if quick_terms_text.strip():
                quick_terms = parse_quick_terms(quick_terms_text)
                if quick_terms:
                    extra_dictionary = {
                        'high_risk': quick_terms,
                        'medium_risk': [],
                        'context_phrases': [],
                        'work_context': []
                    }
                    st.success(f"✅ {len(quick_terms)} term(s) loaded: {', '.join(quick_terms[:5])}{'...' if len(quick_terms) > 5 else ''}")

        with tab2:
            uploaded_dict = st.file_uploader(
                "Upload a CSV/TXT with additional terms",
                type=['csv', 'txt'],
                help="Format: term,category per line. Added to selected categories above."
            )
            if uploaded_dict:
                extra_dictionary = load_dictionary_from_file(uploaded_dict)
                if extra_dictionary:
                    st.success("✅ Additional terms loaded successfully")
                else:
                    st.error("❌ Error loading terms file")

        dictionaries_to_merge = [
            load_predefined_dictionary(PREDEFINED_DICTIONARY_FILES[category])
            for category in selected_categories
        ]

        if extra_dictionary:
            dictionaries_to_merge.append(extra_dictionary)

        dictionary = merge_dictionaries(dictionaries_to_merge) if dictionaries_to_merge else None

        if selected_categories:
            detection_type = " + ".join(selected_categories)
        elif extra_dictionary and dictionary:
            detection_type = "Custom"
        else:
            detection_type = None

        st.subheader("🎚️ Sensitivity")
        sensitivity = st.select_slider(
            "Sensitivity Level",
            options=['low', 'medium', 'high'],
            value='medium',
            help="Low: Fewer false positives | High: Detects more cases"
        )

        custom_threshold = st.slider(
            "Custom Threshold (Optional)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0.0 = Very sensitive | 1.0 = Very strict"
        )

        use_custom = st.checkbox("Use custom threshold")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Chat File")
        uploaded_file = st.file_uploader(
            "Select WhatsApp chat file (.txt)",
            type=['txt'],
            help="Must be a WhatsApp chat export without media"
        )

    with col2:
        if dictionary:
            st.header("📊 Dictionary Summary")
            st.metric("High Risk", len(dictionary['high_risk']))
            st.metric("Medium Risk", len(dictionary['medium_risk']))
            st.metric("Context Phrases", len(dictionary['context_phrases']))
            st.metric("Work Context", len(dictionary['work_context']))
        else:
            st.info("👈 Select at least one category or upload custom terms in the sidebar")

    if uploaded_file and dictionary:
        try:
            content = uploaded_file.read().decode('utf-8')

            is_valid, validation_message = validate_whatsapp_file(content)
            if not is_valid:
                st.error(f"❌ {validation_message}")
                return
            st.success(f"✅ {validation_message}")

            messages = extract_messages_from_text(content)

            if not messages:
                st.error("❌ Could not extract messages from file. Verify it is a valid WhatsApp export.")
                return

            st.success(f"✅ Found {len(messages)} messages")

            dates, senders = extract_dates_and_senders(messages)

            if "filter_date_from" not in st.session_state:
                st.session_state.filter_date_from = dates[0] if dates else None
            if "filter_date_to" not in st.session_state:
                st.session_state.filter_date_to = dates[-1] if dates else None
            if "filter_senders" not in st.session_state:
                st.session_state.filter_senders = senders if senders else []

            st.write("**🗓️ Date Range (Optional)**")
            if dates:
                st.caption(f"Chat contains messages from {dates[0]} to {dates[-1]}")

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                st.session_state.filter_date_from = st.date_input(
                    "From",
                    value=st.session_state.filter_date_from,
                    key="date_from_input"
                )
            with col_date2:
                st.session_state.filter_date_to = st.date_input(
                    "To",
                    value=st.session_state.filter_date_to,
                    key="date_to_input"
                )

            with st.sidebar.expander("🔍 Advanced Filters (Optional)", expanded=True):
                if senders:
                    st.write("**Senders**")
                    st.session_state.filter_senders = st.multiselect(
                        "Select senders to analyze (if none selected, all are analyzed)",
                        options=senders,
                        default=st.session_state.filter_senders,
                        key="senders_filter_input"
                    )

            messages_filtered = messages

            if st.session_state.filter_date_from and st.session_state.filter_date_to:
                filtered_messages = []
                for ts, sender, msg in messages_filtered:
                    msg_date = parse_whatsapp_date(ts.split()[0].strip(','))
                    if msg_date and st.session_state.filter_date_from <= msg_date <= st.session_state.filter_date_to:
                        filtered_messages.append((ts, sender, msg))
                messages_filtered = filtered_messages

            if st.session_state.filter_senders:
                messages_filtered = [
                    (ts, sender, msg) for ts, sender, msg in messages_filtered
                    if sender in st.session_state.filter_senders
                ]

            if len(messages_filtered) < len(messages):
                st.info(f"📊 Filters applied: {len(messages_filtered)} of {len(messages)} messages")

            config = setup_sensitivity(
                sensitivity,
                custom_threshold if use_custom else None
            )

            st.info(f"🎯 Detecting: **{detection_type}** with **{sensitivity}** sensitivity (threshold: {config['threshold']:.2f})")

            with st.spinner("🔄 Analyzing messages..."):
                results = []
                progress_bar = st.progress(0)

                for i, (timestamp, sender, message) in enumerate(messages_filtered):
                    risk, label, words = analyze_message(message, sender, config, dictionary)

                    results.append({
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message,
                        'risk_score': round(risk, 4),
                        'label': label,
                        'detected_words': ', '.join(words) if words else ""
                    })

                    progress_bar.progress((i + 1) / len(messages_filtered))

                progress_bar.empty()

            results_df = pd.DataFrame(results)

            st.header("📈 Analysis Results")

            total_messages = len(results_df)
            detected_messages = len(results_df[results_df['label'] == 'DETECTED'])
            percentage = (detected_messages / total_messages) * 100 if total_messages > 0 else 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📝 Total Messages", total_messages)

            with col2:
                st.metric("🚨 Detected", detected_messages)

            with col3:
                st.metric("📊 Percentage", f"{percentage:.2f}%")

            with col4:
                avg_risk = results_df['risk_score'].mean()
                st.metric("⚖️ Average Risk", f"{avg_risk:.3f}")

            for level, message in generate_smart_alerts(results_df):
                getattr(st, level)(message)

            if detected_messages > 0:
                st.header("📊 Visualizations")
                create_visualizations(results_df, detection_type)

                st.header("🔍 Evidence Found")

                detected_df = results_df[results_df['label'] == 'DETECTED'].copy()
                detected_df = detected_df.sort_values('risk_score', ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    sender_filter = st.multiselect(
                        "Filter by sender:",
                        options=detected_df['sender'].unique(),
                        default=detected_df['sender'].unique()
                    )

                with col2:
                    risk_threshold = st.slider(
                        "Minimum risk to show:",
                        min_value=0.0,
                        max_value=1.0,
                        value=config['threshold'],
                        step=0.05
                    )

                with col3:
                    word_filter = st.text_input(
                        "Search word:",
                        placeholder="Ex: secret, boss..."
                    )

                filtered_df = detected_df[
                    (detected_df['sender'].isin(sender_filter)) &
                    (detected_df['risk_score'] >= risk_threshold)
                ]

                if word_filter:
                    filtered_df = filtered_df[
                        filtered_df['message'].str.contains(word_filter, case=False, na=False) |
                        filtered_df['detected_words'].str.contains(word_filter, case=False, na=False)
                    ]

                st.dataframe(
                    filtered_df[['timestamp', 'sender', 'message', 'risk_score', 'detected_words']].head(50),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "timestamp": st.column_config.TextColumn("📅 Date/Time"),
                        "sender": st.column_config.TextColumn("👤 Sender"),
                        "message": st.column_config.TextColumn("💬 Message", width="large"),
                        "risk_score": st.column_config.NumberColumn("⚖️ Risk", format="%.3f"),
                        "detected_words": st.column_config.TextColumn("🎯 Terms"),
                    }
                )

                if len(filtered_df) > 50:
                    st.info(f"Showing first 50 of {len(filtered_df)} evidence items")

            else:
                st.success("✅ No suspicious patterns detected in the conversation")

            st.header("💾 Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📄 Download Complete CSV",
                    data=csv_data,
                    file_name=f"analysis_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                if detected_messages > 0:
                    detected_csv = io.StringIO()
                    detected_df.to_csv(detected_csv, index=False, encoding='utf-8')
                    detected_data = detected_csv.getvalue()

                    st.download_button(
                        label="🚨 Download Detections Only",
                        data=detected_data,
                        file_name=f"detections_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col3:
                report = f"""EXECUTIVE REPORT - WHATSAPP ANALYZER
{'=' * 50}
Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analyzed file: {uploaded_file.name}
Detection type: {detection_type}
Sensitivity: {sensitivity}
Threshold used: {config['threshold']:.3f}

SUMMARY
{'=' * 50}
Total messages analyzed: {total_messages}
Detected messages: {detected_messages}
Detection percentage: {percentage:.2f}%
Average risk: {avg_risk:.4f}

DISCLAIMER
{'=' * 50}
This report is a support tool, not definitive legal evidence.
Results require manual validation.
"""
                st.download_button(
                    label="📋 Download Executive Report",
                    data=report,
                    file_name=f"report_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Verify that the file is a valid WhatsApp export")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔒 <strong>Privacy:</strong> All files are processed locally. No data is stored.</p>
        <p>⚖️ <strong>Responsible Use:</strong> This tool should be used only for legitimate purposes and respecting privacy.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
