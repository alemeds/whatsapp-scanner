#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALIZADOR DE CONVERSACIONES DE WHATSAPP PARA DETECCIÓN DE PATRONES - STREAMLIT
Aplicación web para detectar diferentes tipos de delitos y comportamientos en chats

Autor: Sistema de Análisis de Comunicaciones
Versión: 3.0 - Streamlit Edition
"""

import streamlit as st
import pandas as pd
import re
import csv
import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Analizador WhatsApp - Detector de Patrones",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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

def setup_sensitivity(level, custom_threshold=None):
    """Configura niveles de sensibilidad"""
    configs = {
        'baja': {
            'threshold': 0.75, 'high_weight': 0.6, 'medium_weight': 0.2, 
            'context_weight': 0.4, 'work_weight': 0.3, 'bonus': 0.1
        },
        'media': {
            'threshold': 0.60, 'high_weight': 0.7, 'medium_weight': 0.3,
            'context_weight': 0.5, 'work_weight': 0.4, 'bonus': 0.15
        },
        'alta': {
            'threshold': 0.45, 'high_weight': 0.8, 'medium_weight': 0.4,
            'context_weight': 0.6, 'work_weight': 0.5, 'bonus': 0.2
        }
    }
    
    config = configs.get(level, configs['media'])
    if custom_threshold is not None:
        config['threshold'] = custom_threshold
    return config

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

# Mapeo de categorías predefinidas a su archivo CSV (vive junto al script)
PREDEFINED_DICTIONARY_FILES = {
    "Acoso Sexual": "diccionario.csv",
    "Cyberbullying": "diccionario_cyberbullying.csv",
    "Amenazas y Violencia": "diccionario_amenazas_violencia.csv",
    "Drogas": "diccionario_drogas.csv",
    "Infidelidad": "diccionario_infidelidad.csv",
    "Malas Palabras (Argentina)": "diccionario_malas_palabras_ARG.csv",
    "Robo y Estafas": "diccionario_robo_estafas.csv",
    "Suicidio y Autolesión": "diccionario_suicidio.csv",
    "Completo (todas las categorías)": "diccionario_completo.csv",
}


def parse_dictionary_rows(reader):
    """Convierte filas (término, categoría) en un diccionario clasificado por riesgo"""
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
    """Parsea términos sueltos separados por coma o salto de línea.
    Un término entre comillas se busca como frase exacta; si no, cada
    término separado por coma se busca de forma independiente (OR)."""
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
    """Carga diccionario desde archivo subido por el usuario"""
    try:
        content = uploaded_file.read().decode('utf-8-sig')

        if uploaded_file.name.endswith('.csv'):
            reader = csv.reader(io.StringIO(content))
        else:  # .txt
            lines = content.strip().split('\n')
            reader = [line.split(',') for line in lines if ',' in line]

        return parse_dictionary_rows(reader)

    except Exception as e:
        st.error(f"Error al cargar diccionario: {e}")
        return None


@st.cache_data
def load_predefined_dictionary(filename):
    """Carga un diccionario predefinido desde el CSV ubicado junto al script"""
    path = Path(__file__).parent / filename
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        return parse_dictionary_rows(reader)


def merge_dictionaries(dictionaries):
    """Combina varios diccionarios en uno solo, sin duplicar términos"""
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

# Caracteres de formato invisibles que agregan los exports de WhatsApp en iPhone
# (marcas de dirección de texto bidireccional) y espacios no estándar
INVISIBLE_CHARS = ['‎', '‏', '‪', '‫', '‬', '‭', '‮']


def normalize_whatsapp_text(content):
    """Quita marcas Unicode invisibles de los exports de iPhone y normaliza espacios no estándar"""
    for char in INVISIBLE_CHARS:
        content = content.replace(char, '')
    return content.replace(' ', ' ').replace('\xa0', ' ')


def validate_whatsapp_file(content):
    """Valida si el contenido parece una exportación de WhatsApp antes de procesarlo"""
    content = normalize_whatsapp_text(content)
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*\d{1,2}:\d{2}.*[-\]].*:'
    total_lines = len(content.split('\n'))
    matches = len(re.findall(pattern, content))
    confidence = matches / max(total_lines, 1)

    if confidence > 0.1:
        return True, f"Formato WhatsApp detectado (confianza: {confidence:.0%})"
    return False, f"No parece ser un archivo de WhatsApp válido (confianza: {confidence:.0%})"


# Mensajes de sistema de WhatsApp a ignorar (frases completas, no palabras sueltas,
# para no descartar mensajes reales que contengan alguna de estas palabras)
SYSTEM_MESSAGES = [
    '<multimedia omitido>', '<media omitted>',
    'se unió usando el enlace de invitación del grupo',
    'cambió el asunto del grupo a', 'eliminó este mensaje',
    'este mensaje fue eliminado', 'los mensajes y las llamadas están cifrados',
    'changed the subject to', 'this message was deleted',
    'creó este grupo', 'se te añadió al grupo', 'añadió al grupo',
    'imagen omitida', 'audio omitido', 'video omitido', 'documento omitido',
    'gif omitido', 'sticker omitido', 'contacto omitida',
]

# Patrones de inicio de mensaje: (timestamp)(sender): (primera línea del texto)
MESSAGE_START_PATTERNS = [
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s[ap]\.\s?m\.)\]\s([^:]+):\s(.*)$'),
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4}(?:,|\s)\s\d{1,2}:\d{2}(?::\d{2})?(?:\s[APap][Mm])?)\]\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[ap]\.\sm\.)\s-\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}(?:\s[APap][Mm])?)\s-\s([^:]+):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$'),
]


def extract_messages_from_text(content):
    """Extrae mensajes de WhatsApp línea por línea, soportando mensajes multilínea"""
    content = normalize_whatsapp_text(content)
    lines = content.split('\n')

    best_pattern = max(
        MESSAGE_START_PATTERNS,
        key=lambda pattern: sum(1 for line in lines if pattern.match(line))
    )

    raw_messages = []
    current = None
    for line in lines:
        match = best_pattern.match(line)
        if match:
            if current:
                raw_messages.append(current)
            timestamp, sender, text = match.groups()
            current = [timestamp.strip(), sender.strip(), text.rstrip('\r')]
        elif current:
            current[2] += '\n' + line.rstrip('\r')
    if current:
        raw_messages.append(current)

    messages = []
    for timestamp, sender, message in raw_messages:
        message = message.strip()
        if not message or any(sys_msg in message.lower() for sys_msg in SYSTEM_MESSAGES):
            continue
        messages.append((timestamp, sender, message))

    return messages

# Cantidad de coincidencias a partir de la cual una categoría alcanza su peso máximo
SATURATION_CAP = 3

def saturating_ratio(matches, cap=SATURATION_CAP):
    """Convierte una cantidad de coincidencias en una proporción [0, 1], saturando en `cap`"""
    return min(matches, cap) / cap

@lru_cache(maxsize=4096)
def _term_pattern(term):
    """Compila (y cachea) una regex que matchea el término como palabra completa, no como substring"""
    return re.compile(r'\b' + re.escape(term) + r'\b')


def term_matches(term, text_lower):
    """True si `term` aparece como palabra completa en el texto (evita 'cama' dentro de 'cámara')"""
    return _term_pattern(term).search(text_lower) is not None


def analyze_message(text, sender, config, dictionary):
    """Analiza un mensaje individual"""
    text_lower = text.lower()

    # Contar coincidencias (por palabra completa, no por substring)
    high_matches = sum(1 for term in dictionary['high_risk'] if term_matches(term, text_lower))
    medium_matches = sum(1 for term in dictionary['medium_risk'] if term_matches(term, text_lower))
    context_matches = sum(1 for phrase in dictionary['context_phrases'] if term_matches(phrase, text_lower))
    work_matches = sum(1 for term in dictionary['work_context'] if term_matches(term, text_lower))

    # Calcular puntuaciones a partir de la cantidad absoluta de coincidencias
    # (saturando a partir de SATURATION_CAP), nunca del tamaño del diccionario:
    # un diccionario con 600 términos no debe diluir el riesgo frente a uno de 20.
    high_score = saturating_ratio(high_matches) * config['high_weight']
    medium_score = saturating_ratio(medium_matches) * config['medium_weight']
    context_score = saturating_ratio(context_matches) * config['context_weight']
    work_score = saturating_ratio(work_matches) * config['work_weight']
    
    # Bonificaciones por combinaciones
    bonus = 0
    if high_matches > 0 and context_matches > 0:
        bonus += config['bonus'] * 1.5
    if high_matches > 0 and work_matches > 0:
        bonus += config['bonus'] * 2.0
    if medium_matches > 0 and context_matches > 0:
        bonus += config['bonus'] * 1.0
    
    # Puntuación final
    risk_score = min(high_score + medium_score + context_score + work_score + bonus, 1.0)
    
    # Palabras detectadas
    detected_words = []
    for category, terms in dictionary.items():
        for term in terms:
            if term_matches(term, text_lower) and term not in detected_words:
                detected_words.append(term)
    
    label = "DETECTADO" if risk_score > config['threshold'] else "NO DETECTADO"
    return risk_score, label, detected_words

def generate_smart_alerts(results_df):
    """Genera alertas basadas en la frecuencia de detecciones y su concentración por remitente"""
    alerts = []
    detected_df = results_df[results_df['label'] == 'DETECTADO']

    if len(detected_df) == 0:
        return alerts

    detection_rate = len(detected_df) / len(results_df)
    if detection_rate > 0.3:
        alerts.append(('error', f"⚠️ CRÍTICO: {len(detected_df)} detecciones de {len(results_df)} mensajes ({detection_rate:.1%})"))
    elif detection_rate > 0.15:
        alerts.append(('warning', f"🔔 ADVERTENCIA: {len(detected_df)} detecciones de {len(results_df)} mensajes ({detection_rate:.1%})"))

    sender_counts = detected_df['sender'].value_counts()
    dominant_pct = sender_counts.iloc[0] / len(detected_df)
    if dominant_pct > 0.7:
        alerts.append(('info', f"👤 REMITENTE DOMINANTE: {sender_counts.index[0]} concentra el {dominant_pct:.1%} de las detecciones"))

    return alerts


def create_visualizations(results_df, detection_type):
    """Crea visualizaciones de los resultados"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de distribución de riesgo
        fig_hist = px.histogram(
            results_df, 
            x='risk_score', 
            nbins=20,
            title=f'Distribución de Puntuación de Riesgo - {detection_type}',
            labels={'risk_score': 'Puntuación de Riesgo', 'count': 'Cantidad de Mensajes'}
        )
        fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red", 
                          annotation_text="Umbral por defecto")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Gráfico de detecciones por remitente
        detections_by_sender = results_df[results_df['label'] == 'DETECTADO']['sender'].value_counts()
        if not detections_by_sender.empty:
            fig_pie = px.pie(
                values=detections_by_sender.values,
                names=detections_by_sender.index,
                title=f'Detecciones por Remitente - {detection_type}'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No se encontraron detecciones para mostrar")

def parse_whatsapp_date(date_str):
    """Parsea una fecha de WhatsApp (múltiples formatos) a datetime object"""
    from datetime import datetime

    formats = [
        '%d/%m/%Y',
        '%d/%m/%y',
        '%m/%d/%Y',
        '%m/%d/%y',
    ]

    date_str = date_str.strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def extract_dates_and_senders(messages):
    """Extrae las fechas únicas y remitentes únicos de los mensajes"""
    dates = set()
    senders = set()

    for timestamp, sender, message in messages:
        date_part = timestamp.split()[0] if timestamp else None
        if date_part:
            parsed_date = parse_whatsapp_date(date_part)
            if parsed_date:
                dates.add(parsed_date)
        senders.add(sender)

    return sorted(dates), sorted(senders)


def get_instructions_text():
    return """# 🔍 Analizador de Conversaciones WhatsApp — Guía de Uso

## ¿Qué es esta herramienta?
Sistema de detección local para analizar patrones en conversaciones de WhatsApp.
Procesa archivos completos sin enviarlos a servidores externos — todo corre en tu máquina.

## Cómo usar

### 1. Preparar el chat
- Abre WhatsApp y ve al chat que querés analizar
- Más opciones (⋮) → Exportar chat → Sin archivos multimedia
- Guarda el archivo (.txt) en tu computadora

### 2. Seleccionar categorías o términos
- **Categorías predefinidas:** Acoso Sexual, Cyberbullying, Amenazas, Drogas, Infidelidad, etc.
- **Términos propios:** Tipea nombres o palabras en la pestaña "📝 Tipear"
  - Separalos por comas o saltos de línea: `juan, pedro, marla`
  - Para buscar frases exactas, usalas entre comillas: `"buenos días"`
- **Archivo CSV:** Si tenes muchos términos, sube un CSV con `término,categoría` por línea

### 3. Ajustar sensibilidad
- **Baja:** Menos falsos positivos, pero puede perder casos reales
- **Media:** Balance recomendado
- **Alta:** Detecta más casos, pero puede tener más falsos positivos

### 4. Subir el chat y analizar
- Sube el archivo .txt que exportaste
- La app analiza automáticamente y muestra resultados

## Resultados
- **DETECTADO:** Mensaje superó el umbral de riesgo
- **NO DETECTADO:** Por debajo del umbral
- **Palabras detectadas:** Cuáles términos se encontraron en cada mensaje

## Privacidad
✅ Todos los archivos se procesan localmente en tu máquina
✅ Nada se almacena ni envía a servidores externos
✅ Los resultados solo los ves tú

## Contacto
Consultas o reportes: antoniolmartinez@gmail.com
"""


def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Analizador de Conversaciones WhatsApp</h1>
        <p>Sistema avanzado para detección de patrones y comportamientos en chats</p>
    </div>
    """, unsafe_allow_html=True)

    # Instrucciones descargables
    with st.expander("📖 Leer instrucciones", expanded=False):
        instructions = get_instructions_text()
        st.markdown(instructions)
        st.download_button(
            label="📥 Descargar instrucciones (TXT)",
            data=instructions,
            file_name="instrucciones_analizador_whatsapp.txt",
            mime="text/plain"
        )

    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Tipo(s) de detección (combinables)
        selected_categories = st.multiselect(
            "🎯 Tipo(s) de Detección",
            options=list(PREDEFINED_DICTIONARY_FILES.keys()),
            help="Podés combinar varias categorías a la vez"
        )

        # Términos puntuales adicionales (opcional, se suman a lo seleccionado arriba)
        st.subheader("📁 Agregar Términos Puntuales (Opcional)")

        tab1, tab2 = st.tabs(["📝 Tipear", "📤 Archivo"])

        extra_dictionary = None

        with tab1:
            quick_terms_text = st.text_area(
                "Escribí nombres o palabras (separadas por comas o saltos de línea)",
                placeholder='juan, pedro\nmarla\n"buenos días"',
                height=80,
                help='Cada término se busca de forma independiente (OR). Entre comillas se busca como frase exacta.'
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
                    st.success(f"✅ {len(quick_terms)} término(s) cargado(s): {', '.join(quick_terms[:5])}{'...' if len(quick_terms) > 5 else ''}")

        with tab2:
            uploaded_dict = st.file_uploader(
                "Subí un CSV/TXT con términos adicionales",
                type=['csv', 'txt'],
                help="Formato: término,categoría por línea. Se suma a las categorías elegidas arriba."
            )
            if uploaded_dict:
                extra_dictionary = load_dictionary_from_file(uploaded_dict)
                if extra_dictionary:
                    st.success("✅ Términos adicionales cargados correctamente")
                else:
                    st.error("❌ Error al cargar el archivo de términos adicionales")

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
            detection_type = "Personalizado"
        else:
            detection_type = None
        
        # Configuración de sensibilidad
        st.subheader("🎚️ Sensibilidad")
        sensitivity = st.select_slider(
            "Nivel de Sensibilidad",
            options=['baja', 'media', 'alta'],
            value='media',
            help="Baja: Menos falsos positivos | Alta: Detecta más casos"
        )
        
        custom_threshold = st.slider(
            "Umbral Personalizado (Opcional)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0.0 = Muy sensible | 1.0 = Muy estricto"
        )
        
        use_custom = st.checkbox("Usar umbral personalizado")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Subir Archivo de Chat")
        uploaded_file = st.file_uploader(
            "Selecciona el archivo de chat de WhatsApp (.txt)",
            type=['txt'],
            help="Debe ser una exportación de chat de WhatsApp sin archivos multimedia"
        )
    
    with col2:
        if dictionary:
            st.header("📊 Resumen del Diccionario")
            st.metric("Alto Riesgo", len(dictionary['high_risk']))
            st.metric("Riesgo Medio", len(dictionary['medium_risk']))
            st.metric("Frases Contexto", len(dictionary['context_phrases']))
            st.metric("Contexto Trabajo", len(dictionary['work_context']))
        else:
            st.info("👈 Elegí al menos una categoría o subí términos puntuales en la barra lateral")
    
    # Procesar archivo si está disponible
    if uploaded_file and dictionary:
        try:
            # Leer contenido del archivo
            content = uploaded_file.read().decode('utf-8')

            # Validar formato antes de procesar
            is_valid, validation_message = validate_whatsapp_file(content)
            if not is_valid:
                st.error(f"❌ {validation_message}")
                return
            st.success(f"✅ {validation_message}")

            # Extraer mensajes
            messages = extract_messages_from_text(content)

            if not messages:
                st.error("❌ No se pudieron extraer mensajes del archivo. Verifica que sea una exportación válida de WhatsApp.")
                return

            st.success(f"✅ Se encontraron {len(messages)} mensajes")

            # Extraer fechas y remitentes para los filtros
            dates, senders = extract_dates_and_senders(messages)

            # Inicializar session_state para los filtros si no existen
            if "use_date_filter" not in st.session_state:
                st.session_state.use_date_filter = False
            if "filter_date_from" not in st.session_state:
                st.session_state.filter_date_from = dates[0] if dates else None
            if "filter_date_to" not in st.session_state:
                st.session_state.filter_date_to = dates[-1] if dates else None
            if "filter_senders" not in st.session_state:
                st.session_state.filter_senders = senders if senders else []

            # Filtros dinámicos basados en el contenido del chat
            with st.sidebar.expander("🔍 Filtros Avanzados (Opcional)", expanded=False):
                # Filtro de fechas
                if dates:
                    st.write("**Rango de Fechas**")
                    st.session_state.use_date_filter = st.checkbox(
                        "🗓️ Filtrar por rango de fechas",
                        value=st.session_state.use_date_filter,
                        key="use_date_filter_input"
                    )

                    if st.session_state.use_date_filter:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.filter_date_from = st.date_input(
                                "Desde",
                                value=st.session_state.filter_date_from,
                                min_value=dates[0],
                                max_value=dates[-1],
                                key="date_from_input"
                            )
                        with col2:
                            st.session_state.filter_date_to = st.date_input(
                                "Hasta",
                                value=st.session_state.filter_date_to,
                                min_value=dates[0],
                                max_value=dates[-1],
                                key="date_to_input"
                            )
                    else:
                        st.info("📅 Analizando TODO el período del chat")

                # Filtro de remitentes
                if senders:
                    st.write("**Remitentes**")
                    st.session_state.filter_senders = st.multiselect(
                        "Selecciona quiénes analizar (si no seleccionas, se analizan todos)",
                        options=senders,
                        default=st.session_state.filter_senders,
                        key="senders_filter_input"
                    )

            # Aplicar filtros a los mensajes
            messages_filtered = messages

            if st.session_state.use_date_filter and dates and st.session_state.filter_date_from and st.session_state.filter_date_to:
                messages_filtered = [
                    (ts, sender, msg) for ts, sender, msg in messages_filtered
                    if (parse_whatsapp_date(ts.split()[0]) and
                        st.session_state.filter_date_from <= parse_whatsapp_date(ts.split()[0]) <= st.session_state.filter_date_to)
                ]

            if st.session_state.filter_senders:
                messages_filtered = [
                    (ts, sender, msg) for ts, sender, msg in messages_filtered
                    if sender in st.session_state.filter_senders
                ]

            # Mostrar si se aplicaron filtros
            if len(messages_filtered) < len(messages):
                st.info(f"📊 Filtros aplicados: {len(messages_filtered)} de {len(messages)} mensajes")

            # Configurar análisis
            config = setup_sensitivity(
                sensitivity,
                custom_threshold if use_custom else None
            )

            # Mostrar configuración
            st.info(f"🎯 Detectando: **{detection_type}** con sensibilidad **{sensitivity}** (umbral: {config['threshold']:.2f})")

            # Procesar mensajes
            with st.spinner("🔄 Analizando mensajes..."):
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
            
            # Crear DataFrame
            results_df = pd.DataFrame(results)
            
            # Mostrar estadísticas
            st.header("📈 Resultados del Análisis")
            
            total_messages = len(results_df)
            detected_messages = len(results_df[results_df['label'] == 'DETECTADO'])
            percentage = (detected_messages / total_messages) * 100 if total_messages > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Total Mensajes", total_messages)
            
            with col2:
                st.metric("🚨 Detectados", detected_messages)
            
            with col3:
                st.metric("📊 Porcentaje", f"{percentage:.2f}%")
            
            with col4:
                avg_risk = results_df['risk_score'].mean()
                st.metric("⚖️ Riesgo Promedio", f"{avg_risk:.3f}")

            # Alertas inteligentes
            for level, message in generate_smart_alerts(results_df):
                getattr(st, level)(message)

            # Mostrar visualizaciones
            if detected_messages > 0:
                st.header("📊 Visualizaciones")
                create_visualizations(results_df, detection_type)
                
                # Mostrar evidencias
                st.header("🔍 Evidencias Encontradas")
                
                detected_df = results_df[results_df['label'] == 'DETECTADO'].copy()
                detected_df = detected_df.sort_values('risk_score', ascending=False)
                
                # Filtros
                col1, col2, col3 = st.columns(3)
                with col1:
                    sender_filter = st.multiselect(
                        "Filtrar por remitente:",
                        options=detected_df['sender'].unique(),
                        default=detected_df['sender'].unique()
                    )

                with col2:
                    risk_threshold = st.slider(
                        "Riesgo mínimo a mostrar:",
                        min_value=0.0,
                        max_value=1.0,
                        value=config['threshold'],
                        step=0.05
                    )

                with col3:
                    word_filter = st.text_input(
                        "Buscar palabra:",
                        placeholder="Ej: secreto, jefe..."
                    )

                # Aplicar filtros
                filtered_df = detected_df[
                    (detected_df['sender'].isin(sender_filter)) &
                    (detected_df['risk_score'] >= risk_threshold)
                ]

                if word_filter:
                    filtered_df = filtered_df[
                        filtered_df['message'].str.contains(word_filter, case=False, na=False) |
                        filtered_df['detected_words'].str.contains(word_filter, case=False, na=False)
                    ]

                # Mostrar evidencias filtradas en tabla
                st.dataframe(
                    filtered_df[['timestamp', 'sender', 'message', 'risk_score', 'detected_words']].head(50),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "timestamp": st.column_config.TextColumn("📅 Fecha/Hora"),
                        "sender": st.column_config.TextColumn("👤 Remitente"),
                        "message": st.column_config.TextColumn("💬 Mensaje", width="large"),
                        "risk_score": st.column_config.NumberColumn("⚖️ Riesgo", format="%.3f"),
                        "detected_words": st.column_config.TextColumn("🎯 Términos"),
                    }
                )

                if len(filtered_df) > 50:
                    st.info(f"Mostrando las primeras 50 evidencias de {len(filtered_df)} encontradas")
            
            else:
                st.success("✅ No se detectaron patrones sospechosos en la conversación")
            
            # Opción de descarga
            st.header("💾 Descargar Resultados")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                # Descargar CSV completo
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📄 Descargar CSV Completo",
                    data=csv_data,
                    file_name=f"analisis_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                # Descargar solo detecciones
                if detected_messages > 0:
                    detected_csv = io.StringIO()
                    detected_df.to_csv(detected_csv, index=False, encoding='utf-8')
                    detected_data = detected_csv.getvalue()

                    st.download_button(
                        label="🚨 Descargar Solo Detecciones",
                        data=detected_data,
                        file_name=f"detecciones_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col3:
                # Reporte ejecutivo en texto
                report = f"""REPORTE EJECUTIVO - ANALIZADOR DE WHATSAPP
{'=' * 50}
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Archivo analizado: {uploaded_file.name}
Tipo de detección: {detection_type}
Sensibilidad: {sensitivity}
Umbral usado: {config['threshold']:.3f}

RESUMEN
{'=' * 50}
Total de mensajes analizados: {total_messages}
Mensajes detectados: {detected_messages}
Porcentaje de detección: {percentage:.2f}%
Riesgo promedio: {avg_risk:.4f}

DISCLAIMER
{'=' * 50}
Este reporte es una herramienta de apoyo, no constituye evidencia legal
definitiva. Los resultados requieren validación manual.
"""
                st.download_button(
                    label="📋 Descargar Reporte Ejecutivo",
                    data=report,
                    file_name=f"reporte_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.info("💡 Verifica que el archivo sea una exportación válida de WhatsApp")
    
    # Footer con información
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔒 <strong>Privacidad:</strong> Todos los archivos se procesan localmente. No se almacenan datos.</p>
        <p>⚖️ <strong>Uso Responsable:</strong> Esta herramienta debe usarse únicamente con fines legítimos y respetando la privacidad.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()