#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""WhatsApp Analyzer - Pattern Detection Application."""

import streamlit as st
import pandas as pd
import io
from datetime import datetime

from .parser import extract_messages_from_text, validate_whatsapp_file, extract_dates_and_senders, parse_whatsapp_date
from .dictionary import PREDEFINED_DICTIONARY_FILES, load_predefined_dictionary, load_dictionary_from_file, parse_quick_terms, merge_dictionaries
from .analyzer import analyze_message, setup_sensitivity
from .ui import generate_smart_alerts, create_visualizations, get_instructions_text
from .i18n import t, get_language, set_language, SUPPORTED_LANGUAGES

st.set_page_config(
    page_title=t("app.page_title"),
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
    with st.sidebar:
        st.write("**Language / Idioma**")
        selected_lang = st.selectbox(
            "Choose language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0 if get_language() == "es" else 1,
            key="language_selector"
        )
        set_language(SUPPORTED_LANGUAGES[selected_lang])

    st.markdown(f"""
    <div class="main-header">
        <h1>🔍 {t('app.title')}</h1>
        <p>{t('app.description')}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📖 {t('sidebar.configuration')}", expanded=False):
        instructions = get_instructions_text()
        st.markdown(instructions)
        st.download_button(
            label="📥 Download instructions (TXT)",
            data=instructions,
            file_name="whatsapp_analyzer_instructions.txt",
            mime="text/plain"
        )

    with st.sidebar:
        st.header(f"⚙️ {t('sidebar.configuration')}")

        selected_categories = st.multiselect(
            t("sidebar.detection_type"),
            options=list(PREDEFINED_DICTIONARY_FILES.keys()),
            help=t("sidebar.detection_type_help")
        )

        st.subheader(f"📁 {t('sidebar.add_custom_terms')}")

        tab1, tab2 = st.tabs([f"📝 {t('tabs.type')}", f"📤 {t('tabs.file')}"])

        extra_dictionary = None

        with tab1:
            quick_terms_text = st.text_area(
                t("custom_terms.type_placeholder"),
                placeholder=t("custom_terms.type_placeholder"),
                height=80,
                help=t("custom_terms.type_help")
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
                t("custom_terms.file_uploader"),
                type=['csv', 'txt'],
                help=t("custom_terms.file_uploader_help")
            )
            if uploaded_dict:
                extra_dictionary = load_dictionary_from_file(uploaded_dict)
                if extra_dictionary:
                    st.success(f"✅ {t('custom_terms.success')}")
                else:
                    st.error(f"❌ {t('custom_terms.error')}")

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

        st.subheader(f"🎚️ {t('sidebar.sensitivity')}")
        sensitivity = st.select_slider(
            t("sidebar.sensitivity"),
            options=['low', 'medium', 'high'],
            value='medium',
            help=t("sidebar.sensitivity_help")
        )

        custom_threshold = st.slider(
            t("sidebar.custom_threshold"),
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help=t("sidebar.custom_threshold_help")
        )

        use_custom = st.checkbox(t("sidebar.use_custom_threshold"))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"📤 {t('upload.header')}")
        uploaded_file = st.file_uploader(
            t("upload.label"),
            type=['txt'],
            help=t("upload.help")
        )

    with col2:
        if dictionary:
            st.header(f"📊 {t('upload.dictionary_summary')}")
            st.metric(t("dictionary.high_risk"), len(dictionary['high_risk']))
            st.metric(t("dictionary.medium_risk"), len(dictionary['medium_risk']))
            st.metric(t("dictionary.context_phrases"), len(dictionary['context_phrases']))
            st.metric(t("dictionary.work_context"), len(dictionary['work_context']))
        else:
            st.info(f"👈 {t('upload.no_dictionary')}")

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
                st.error(f"❌ {t('file_processing.error')}")
                return

            st.success(f"✅ {t('file_processing.success', count=len(messages))}")

            dates, senders = extract_dates_and_senders(messages)

            if "filter_date_from" not in st.session_state:
                st.session_state.filter_date_from = dates[0] if dates else None
            if "filter_date_to" not in st.session_state:
                st.session_state.filter_date_to = dates[-1] if dates else None
            if "filter_senders" not in st.session_state:
                st.session_state.filter_senders = senders if senders else []

            st.write(f"**🗓️ {t('filters.date_range')}**")
            if dates:
                st.caption(t("filters.chat_contains", from_=dates[0], to=dates[-1]))

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                st.session_state.filter_date_from = st.date_input(
                    t("filters.from"),
                    value=st.session_state.filter_date_from,
                    key="date_from_input"
                )
            with col_date2:
                st.session_state.filter_date_to = st.date_input(
                    t("filters.to"),
                    value=st.session_state.filter_date_to,
                    key="date_to_input"
                )

            with st.sidebar.expander(f"🔍 {t('sidebar.advanced_filters')}", expanded=True):
                if senders:
                    st.write(f"**{t('sidebar.senders')}**")
                    st.session_state.filter_senders = st.multiselect(
                        t("sidebar.select_senders"),
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
                st.info(t("filters.applied", filtered=len(messages_filtered), total=len(messages)))

            config = setup_sensitivity(
                sensitivity,
                custom_threshold if use_custom else None
            )

            st.info(f"🎯 {t('analysis.detecting', detection_type=detection_type, sensitivity=sensitivity, threshold=config['threshold'])}")

            with st.spinner(f"🔄 {t('analysis.analyzing')}"):
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

            st.header(f"📈 {t('analysis.header')}")

            total_messages = len(results_df)
            detected_messages = len(results_df[results_df['label'] == 'DETECTED'])
            percentage = (detected_messages / total_messages) * 100 if total_messages > 0 else 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(f"📝 {t('analysis.total_messages')}", total_messages)

            with col2:
                st.metric(f"🚨 {t('analysis.detected')}", detected_messages)

            with col3:
                st.metric(f"📊 {t('analysis.percentage')}", f"{percentage:.2f}%")

            with col4:
                avg_risk = results_df['risk_score'].mean()
                st.metric(f"⚖️ {t('analysis.average_risk')}", f"{avg_risk:.3f}")

            for level, message in generate_smart_alerts(results_df):
                getattr(st, level)(message)

            if detected_messages > 0:
                st.header(f"📊 {t('analysis.visualizations')}")
                create_visualizations(results_df, detection_type)

                st.header(f"🔍 {t('analysis.evidence')}")

                detected_df = results_df[results_df['label'] == 'DETECTED'].copy()
                detected_df = detected_df.sort_values('risk_score', ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    sender_filter = st.multiselect(
                        t("analysis.filter_by_sender"),
                        options=detected_df['sender'].unique(),
                        default=detected_df['sender'].unique()
                    )

                with col2:
                    risk_threshold = st.slider(
                        t("analysis.minimum_risk"),
                        min_value=0.0,
                        max_value=1.0,
                        value=config['threshold'],
                        step=0.05
                    )

                with col3:
                    word_filter = st.text_input(
                        t("analysis.search_word"),
                        placeholder=t("analysis.search_placeholder")
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
                        "timestamp": st.column_config.TextColumn(f"📅 {t('table.date')}"),
                        "sender": st.column_config.TextColumn(f"👤 {t('table.sender')}"),
                        "message": st.column_config.TextColumn(f"💬 {t('table.message')}", width="large"),
                        "risk_score": st.column_config.NumberColumn(f"⚖️ {t('table.risk')}", format="%.3f"),
                        "detected_words": st.column_config.TextColumn(f"🎯 {t('table.terms')}"),
                    }
                )

                if len(filtered_df) > 50:
                    st.info(t("analysis.showing_results", count=len(filtered_df)))

            else:
                st.success(f"✅ {t('analysis.no_patterns')}")

            st.header(f"💾 {t('downloads.header')}")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label=f"📄 {t('downloads.complete_csv')}",
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
                        label=f"🚨 {t('downloads.detections_only')}",
                        data=detected_data,
                        file_name=f"detections_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col3:
                report = f"""{t('report.title')}
{'=' * 50}
{t('report.analysis_date')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{t('report.analyzed_file')}: {uploaded_file.name}
{t('report.detection_type')}: {detection_type}
{t('report.sensitivity')}: {sensitivity}
{t('report.threshold_used')}: {config['threshold']:.3f}

{t('report.summary')}
{'=' * 50}
{t('report.total_messages')}: {total_messages}
{t('report.detected_messages')}: {detected_messages}
{t('report.detection_percentage')}: {percentage:.2f}%
{t('report.average_risk')}: {avg_risk:.4f}

{t('report.disclaimer')}
{'=' * 50}
{t('report.disclaimer_text')}
"""
                st.download_button(
                    label=f"📋 {t('downloads.executive_report')}",
                    data=report,
                    file_name=f"report_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"❌ {t('errors.file_error')}: {str(e)}")
            st.info(f"💡 {t('errors.verify_file')}")

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔒 <strong>{t('footer.privacy').split(':')[0]}:</strong> {t('footer.privacy').split(':')[1]}</p>
        <p>⚖️ <strong>{t('footer.responsible_use').split(':')[0]}:</strong> {t('footer.responsible_use').split(':')[1]}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
