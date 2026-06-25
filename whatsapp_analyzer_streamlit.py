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
    .evidence-card {
        background: #fff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { border-left: 4px solid #dc3545; }
    .risk-medium { border-left: 4px solid #ffc107; }
    .risk-low { border-left: 4px solid #28a745; }
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

def extract_messages_from_text(content):
    """Extrae mensajes de texto de WhatsApp"""
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[ap]\.\sm\.)\s-\s([^:]+):\s(.+)',
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}(?:,|\s)\s\d{1,2}:\d{2}(?::\d{2})?(?:\s[APap][Mm])?)\]\s([^:]+):\s(.+)',
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}(?:\s[APap][Mm])?)\s-\s([^:]+):\s(.+)',
        r'(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches and len(matches) > 5:
            return [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
    
    return []

# Cantidad de coincidencias a partir de la cual una categoría alcanza su peso máximo
SATURATION_CAP = 3

def saturating_ratio(matches, cap=SATURATION_CAP):
    """Convierte una cantidad de coincidencias en una proporción [0, 1], saturando en `cap`"""
    return min(matches, cap) / cap

def analyze_message(text, sender, config, dictionary):
    """Analiza un mensaje individual"""
    text_lower = text.lower()
    
    # Contar coincidencias
    high_matches = sum(1 for term in dictionary['high_risk'] if term in text_lower)
    medium_matches = sum(1 for term in dictionary['medium_risk'] if term in text_lower)
    context_matches = sum(1 for phrase in dictionary['context_phrases'] if phrase in text_lower)
    work_matches = sum(1 for term in dictionary['work_context'] if term in text_lower)

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
            if term in text_lower and term not in detected_words:
                detected_words.append(term)
    
    label = "DETECTADO" if risk_score > config['threshold'] else "NO DETECTADO"
    return risk_score, label, detected_words

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

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Analizador de Conversaciones WhatsApp</h1>
        <p>Sistema avanzado para detección de patrones y comportamientos en chats</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        uploaded_dict = st.file_uploader(
            "Subí un CSV/TXT con términos adicionales",
            type=['csv', 'txt'],
            help="Formato: término,categoría por línea. Se suma a las categorías elegidas arriba."
        )

        dictionaries_to_merge = [
            load_predefined_dictionary(PREDEFINED_DICTIONARY_FILES[category])
            for category in selected_categories
        ]

        if uploaded_dict:
            extra_dictionary = load_dictionary_from_file(uploaded_dict)
            if extra_dictionary:
                dictionaries_to_merge.append(extra_dictionary)
                st.success("✅ Términos adicionales cargados correctamente")
            else:
                st.error("❌ Error al cargar el archivo de términos adicionales")

        dictionary = merge_dictionaries(dictionaries_to_merge) if dictionaries_to_merge else None

        if selected_categories:
            detection_type = " + ".join(selected_categories)
        elif uploaded_dict and dictionary:
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
            
            # Extraer mensajes
            messages = extract_messages_from_text(content)
            
            if not messages:
                st.error("❌ No se pudieron extraer mensajes del archivo. Verifica que sea una exportación válida de WhatsApp.")
                return
            
            st.success(f"✅ Se encontraron {len(messages)} mensajes")
            
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
                
                for i, (timestamp, sender, message) in enumerate(messages):
                    risk, label, words = analyze_message(message, sender, config, dictionary)
                    
                    results.append({
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message,
                        'risk_score': round(risk, 4),
                        'label': label,
                        'detected_words': ', '.join(words) if words else ""
                    })
                    
                    progress_bar.progress((i + 1) / len(messages))
                
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
            
            # Mostrar visualizaciones
            if detected_messages > 0:
                st.header("📊 Visualizaciones")
                create_visualizations(results_df, detection_type)
                
                # Mostrar evidencias
                st.header("🔍 Evidencias Encontradas")
                
                detected_df = results_df[results_df['label'] == 'DETECTADO'].copy()
                detected_df = detected_df.sort_values('risk_score', ascending=False)
                
                # Filtros
                col1, col2 = st.columns(2)
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
                
                # Aplicar filtros
                filtered_df = detected_df[
                    (detected_df['sender'].isin(sender_filter)) &
                    (detected_df['risk_score'] >= risk_threshold)
                ]
                
                # Mostrar evidencias filtradas
                for idx, row in filtered_df.head(20).iterrows():
                    risk_level = "high" if row['risk_score'] > 0.8 else "medium" if row['risk_score'] > 0.6 else "low"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="evidence-card risk-{risk_level}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <strong>👤 {row['sender']}</strong>
                                <span style="color: #666;">📅 {row['timestamp']}</span>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                                💬 {row['message']}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span><strong>⚖️ Riesgo:</strong> {row['risk_score']:.3f}</span>
                                <span><strong>🎯 Términos:</strong> {row['detected_words'] if row['detected_words'] else 'N/A'}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(filtered_df) > 20:
                    st.info(f"Mostrando las primeras 20 evidencias de {len(filtered_df)} encontradas")
            
            else:
                st.success("✅ No se detectaron patrones sospechosos en la conversación")
            
            # Opción de descarga
            st.header("💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
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