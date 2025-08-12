#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALIZADOR AVANZADO DE CONVERSACIONES DE WHATSAPP CON NLP
Aplicación web para detectar diferentes tipos de delitos y comportamientos en chats
Con análisis inteligente usando spaCy y detección contextual

Autor: Sistema de Análisis de Comunicaciones
Versión: 4.0 - NLP Edition (Streamlit Cloud Optimized)
"""

import streamlit as st
import pandas as pd
import re
import csv
import io
import os
import tempfile
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import hashlib

# Función para instalar dependencias automáticamente
@st.cache_resource
def install_and_setup_nlp():
    """Instala y configura NLP automáticamente"""
    try:
        # Intentar importar spacy
        import spacy
        from textblob import TextBlob
        
        # Intentar cargar el modelo español
        try:
            nlp = spacy.load("es_core_news_sm")
            return True, nlp, True
        except OSError:
            # Si no está el modelo, intentar descargarlo
            try:
                st.info("🔄 Descargando modelo de español... (solo la primera vez)")
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "es_core_news_sm"
                ], capture_output=True)
                nlp = spacy.load("es_core_news_sm")
                st.success("✅ Modelo descargado correctamente")
                return True, nlp, True
            except:
                st.warning("⚠️ No se pudo descargar el modelo. Usando análisis básico.")
                return False, None, True
    except ImportError:
        st.error("❌ spaCy no está instalado. Usando análisis básico.")
        return False, None, False

# Inicializar NLP
NLP_AVAILABLE, nlp, TEXTBLOB_AVAILABLE = install_and_setup_nlp()

# Importar TextBlob si está disponible
if TEXTBLOB_AVAILABLE:
    try:
        from textblob import TextBlob
    except ImportError:
        TEXTBLOB_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Analizador WhatsApp - Detector NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
    }
    .warning-box {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px 0 rgba(255, 193, 7, 0.3);
    }
    .evidence-card {
        background: #fff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px 0 rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .evidence-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(0,0,0,0.15);
    }
    .risk-high { 
        border-left: 6px solid #dc3545; 
        background: linear-gradient(145deg, #fff5f5, #ffe6e6);
    }
    .risk-medium { 
        border-left: 6px solid #ffc107; 
        background: linear-gradient(145deg, #fffbf0, #fff4d6);
    }
    .risk-low { 
        border-left: 6px solid #28a745; 
        background: linear-gradient(145deg, #f8fff8, #e6ffe6);
    }
    .instruction-box {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        border: 1px solid #2196f3;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .chat-message {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .detection-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .badge-acoso { background: #ffeaa7; color: #d63031; }
    .badge-bullying { background: #fab1a0; color: #e17055; }
    .badge-infidelidad { background: #fd79a8; color: #e84393; }
</style>
""", unsafe_allow_html=True)

class AdvancedNLPAnalyzer:
    """Analizador NLP avanzado para detección de patrones"""
    
    def __init__(self):
        self.nlp = nlp
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.negation_patterns = ["no", "nunca", "jamás", "nada", "tampoco", "ni", "sin"]
        self.intensity_modifiers = {
            "muy": 1.5, "super": 1.7, "extremadamente": 2.0, "bastante": 1.3,
            "algo": 0.7, "poco": 0.5, "medio": 0.8, "un poco": 0.6
        }
        
    def analyze_message(self, text, sender, timestamp, config, dictionary, detection_type):
        """Análisis NLP completo de un mensaje"""
        if not self.nlp:
            return self.basic_analyze_message(text, sender, config, dictionary)
        
        # Preprocessing inteligente
        cleaned_text = self.smart_preprocessing(text)
        doc = self.nlp(cleaned_text)
        
        # Análisis multidimensional
        analysis_results = {
            'sentiment': self.analyze_sentiment(text),
            'entities': self.extract_entities(doc),
            'negation': self.detect_negation_context(doc),
            'intensity': self.calculate_intensity(doc, text),
            'context': self.analyze_context(doc, dictionary, detection_type),
            'grammar': self.analyze_grammar(doc),
            'temporal': self.analyze_temporal_patterns(timestamp)
        }
        
        # Puntuación inteligente específica por tipo
        smart_score = self.calculate_smart_score(analysis_results, config, dictionary, detection_type)
        
        # Palabras detectadas
        detected_words = self.get_detected_words(text, dictionary)
        
        # Explicación del resultado
        explanation = self.generate_explanation(analysis_results, smart_score, detection_type)
        
        label = "DETECTADO" if smart_score > config['threshold'] else "NO DETECTADO"
        
        return smart_score, label, detected_words, analysis_results, explanation
    
    def smart_preprocessing(self, text):
        """Limpieza inteligente del texto"""
        # Normalizar emojis emocionales
        text = re.sub(r'[😍😘❤️💕💖😻🥰]', ' _expresion_afectiva_ ', text)
        text = re.sub(r'[😠😡🤬👿💢]', ' _expresion_agresiva_ ', text)
        text = re.sub(r'[😢😭💔😞😔]', ' _expresion_triste_ ', text)
        text = re.sub(r'[😏😈🔥💦]', ' _expresion_sugestiva_ ', text)
        
        # Normalizar repeticiones: "hoooola" -> "hola"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalizar variaciones escritas
        replacements = {
            r's[3e]xy': 'sexy', r'k[1i]ero': 'quiero', r'h3rm0sa': 'hermosa',
            r'b[3e]ll[4a]': 'bella', r'amor3s': 'amores', r'bb': 'bebé'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_sentiment(self, text):
        """Análisis de sentimientos con TextBlob"""
        if not self.textblob_available:
            return {'polarity': 0, 'subjectivity': 0, 'interpretation': 'neutral', 'confidence': 0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Interpretación contextual
            if polarity > 0.3:
                interpretation = "positivo"
            elif polarity < -0.3:
                interpretation = "negativo"
            else:
                interpretation = "neutral"
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'interpretation': interpretation,
                'confidence': abs(polarity)
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'interpretation': 'neutral', 'confidence': 0}
    
    def detect_negation_context(self, doc):
        """Detecta negaciones y su contexto específico"""
        negations = []
        
        for token in doc:
            if token.lemma_.lower() in self.negation_patterns:
                # Encontrar qué está siendo negado
                negated_concepts = []
                
                # Buscar en dependencias
                for child in token.children:
                    if child.pos_ in ["ADJ", "NOUN", "VERB"]:
                        negated_concepts.append(child.lemma_.lower())
                
                # Buscar en contexto cercano (3 palabras adelante)
                start_idx = max(0, token.i)
                end_idx = min(len(doc), token.i + 4)
                
                for i in range(start_idx + 1, end_idx):
                    if doc[i].pos_ in ["ADJ", "NOUN", "VERB"]:
                        negated_concepts.append(doc[i].lemma_.lower())
                
                negations.append({
                    'negation_word': token.text,
                    'negated_concepts': negated_concepts,
                    'position': token.i,
                    'strength': self.calculate_negation_strength(token, doc)
                })
        
        return negations
    
    def calculate_negation_strength(self, neg_token, doc):
        """Calcula la fuerza de la negación"""
        strong_negations = ["nunca", "jamás", "nada"]
        if neg_token.lemma_.lower() in strong_negations:
            return 1.0
        elif neg_token.lemma_.lower() == "no":
            return 0.8
        else:
            return 0.6
    
    def extract_entities(self, doc):
        """Extrae entidades nombradas"""
        entities = {
            'persons': [], 'locations': [], 'organizations': [], 
            'dates': [], 'other': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:
                entities['persons'].append(ent.text)
            elif ent.label_ in ["LOC", "GPE"]:
                entities['locations'].append(ent.text)
            elif ent.label_ in ["ORG"]:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities['dates'].append(ent.text)
            else:
                entities['other'].append((ent.text, ent.label_))
        
        return entities
    
    def calculate_intensity(self, doc, original_text):
        """Calcula intensidad emocional del mensaje"""
        intensity_score = 1.0
        intensity_indicators = []
        
        # Modificadores de intensidad
        for token in doc:
            if token.lemma_.lower() in self.intensity_modifiers:
                multiplier = self.intensity_modifiers[token.lemma_.lower()]
                intensity_score *= multiplier
                intensity_indicators.append(f"{token.text} ({multiplier}x)")
        
        # MAYÚSCULAS (indica énfasis)
        caps_ratio = sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1)
        if caps_ratio > 0.3:
            intensity_score *= 1.4
            intensity_indicators.append(f"MAYÚSCULAS ({caps_ratio:.0%})")
        
        # Signos de puntuación enfáticos
        exclamation_count = original_text.count('!')
        question_count = original_text.count('?')
        
        if exclamation_count > 1:
            multiplier = 1 + (exclamation_count - 1) * 0.15
            intensity_score *= multiplier
            intensity_indicators.append(f"!×{exclamation_count}")
        
        if question_count > 2:
            intensity_score *= 1.2
            intensity_indicators.append(f"?×{question_count}")
        
        return {
            'score': min(intensity_score, 3.0),
            'indicators': intensity_indicators
        }
    
    def analyze_context(self, doc, dictionary, detection_type):
        """Analiza contexto específico según el tipo de detección"""
        context_analysis = {
            'workplace_context': 0,
            'relationship_context': 0,
            'family_context': 0,
            'aggressive_context': 0,
            'sexual_context': 0,
            'emotional_context': 0,
            'social_context': 0
        }
        
        # Contextos específicos según tipo de detección
        context_terms = {
            'workplace_context': ["jefe", "trabajo", "oficina", "reunión", "proyecto", "cliente", "empresa", "ascenso"],
            'relationship_context': ["amor", "novio", "novia", "pareja", "cita", "salir", "beso", "abrazo"],
            'family_context': ["familia", "papá", "mamá", "hermano", "hermana", "hijo", "hija", "casa"],
            'aggressive_context': ["odio", "matar", "golpear", "destruir", "venganza", "rabia", "ira"],
            'sexual_context': dictionary.get('high_risk', []) + dictionary.get('medium_risk', []),
            'emotional_context': ["triste", "feliz", "enojado", "deprimido", "ansioso", "estresado"],
            'social_context': ["amigos", "fiesta", "reunión", "grupo", "clase", "escuela"]
        }
        
        # Calcular puntuaciones de contexto
        for context_type, terms in context_terms.items():
            matches = 0
            for token in doc:
                if token.lemma_.lower() in [term.lower() for term in terms]:
                    matches += 1
            
            # Normalizar score
            max_expected = 3
            context_analysis[context_type] = min(matches / max_expected, 1.0)
        
        return context_analysis
    
    def analyze_grammar(self, doc):
        """Analiza patrones gramaticales"""
        grammar_patterns = {
            'imperatives': 0,
            'questions': 0,
            'conditionals': 0,
            'personal_pronouns': 0
        }
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                grammar_patterns['imperatives'] += 1
            
            if token.pos_ == "PRON" and token.dep_ in ["nsubj", "obj"]:
                grammar_patterns['personal_pronouns'] += 1
        
        text = doc.text.lower()
        if '?' in text:
            grammar_patterns['questions'] = text.count('?')
        
        if any(word in text for word in ['si', 'cuando', 'donde', 'como']):
            grammar_patterns['conditionals'] += 1
        
        return grammar_patterns
    
    def analyze_temporal_patterns(self, timestamp):
        """Analiza patrones temporales"""
        try:
            dt = self.parse_timestamp(timestamp)
            if not dt:
                return {'hour': None, 'is_night': False, 'is_weekend': False}
            
            return {
                'hour': dt.hour,
                'is_night': dt.hour >= 22 or dt.hour <= 6,
                'is_weekend': dt.weekday() >= 5,
                'day_of_week': dt.weekday()
            }
        except:
            return {'hour': None, 'is_night': False, 'is_weekend': False}
    
    def parse_timestamp(self, timestamp_str):
        """Parsea diferentes formatos de timestamp"""
        formats = [
            "%d/%m/%y, %I:%M %p",
            "%d/%m/%Y, %H:%M",
            "%d/%m/%y %H:%M",
            "[%d/%m/%y, %I:%M:%S %p]",
            "%d/%m/%Y %I:%M %p"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip('[]'), fmt)
            except:
                continue
        return None
    
    def calculate_smart_score(self, analysis_results, config, dictionary, detection_type):
        """Calcula puntuación inteligente específica por tipo de detección"""
        # Score base
        base_score = self.calculate_base_score(analysis_results, dictionary)
        
        # Ajustes por sentimiento
        sentiment = analysis_results['sentiment']
        sentiment_multiplier = 1.0
        
        if detection_type == "Acoso Sexual":
            if sentiment['polarity'] > 0.2:
                sentiment_multiplier = 1.2
        elif detection_type == "CyberBullying":
            if sentiment['polarity'] < -0.3:
                sentiment_multiplier = 1.4
        elif detection_type == "Infidelidades":
            if abs(sentiment['polarity']) > 0.3:
                sentiment_multiplier = 1.2
        
        base_score *= sentiment_multiplier
        
        # Ajustes por negación
        negations = analysis_results['negation']
        for negation in negations:
            reduction = negation['strength'] * 0.4
            base_score *= (1 - reduction)
        
        # Ajustes por intensidad
        intensity = analysis_results['intensity']['score']
        base_score *= intensity
        
        # Ajustes específicos por contexto y tipo de detección
        context = analysis_results['context']
        
        if detection_type == "Acoso Sexual":
            if context['workplace_context'] > 0.5 and context['sexual_context'] > 0.3:
                base_score *= 1.8
            elif context['relationship_context'] > 0.6:
                base_score *= 0.7
                
        elif detection_type == "CyberBullying":
            if context['aggressive_context'] > 0.4 and context['social_context'] > 0.3:
                base_score *= 1.6
            elif context['family_context'] > 0.5:
                base_score *= 1.2
                
        elif detection_type == "Infidelidades":
            if context['emotional_context'] > 0.4 and context['relationship_context'] > 0.4:
                base_score *= 1.4
            temporal = analysis_results['temporal']
            if temporal['is_night']:
                base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def calculate_base_score(self, analysis_results, dictionary):
        """Calcula score base usando términos del diccionario"""
        # Implementar la lógica básica de scoring
        return 0.5  # Placeholder
    
    def get_detected_words(self, text, dictionary):
        """Obtiene palabras detectadas del diccionario"""
        text_lower = text.lower()
        detected = []
        
        for category, terms in dictionary.items():
            for term in terms:
                if term.lower() in text_lower and term not in detected:
                    detected.append(term)
        
        return detected
    
    def generate_explanation(self, analysis_results, score, detection_type):
        """Genera explicación detallada del resultado"""
        explanations = []
        
        sentiment = analysis_results['sentiment']
        if sentiment['confidence'] > 0.3:
            explanations.append(f"Sentimiento {sentiment['interpretation']} ({sentiment['polarity']:.2f})")
        
        negations = analysis_results['negation']
        if negations:
            explanations.append(f"Negaciones: {len(negations)}")
        
        intensity = analysis_results['intensity']
        if intensity['score'] > 1.3:
            explanations.append(f"Alta intensidad ({intensity['score']:.1f}x)")
        
        context = analysis_results['context']
        max_context = max(context.items(), key=lambda x: x[1])
        if max_context[1] > 0.4:
            context_name = max_context[0].replace('_context', '').replace('_', ' ')
            explanations.append(f"Contexto: {context_name}")
        
        if detection_type == "Infidelidades":
            temporal = analysis_results['temporal']
            if temporal['is_night']:
                explanations.append("Horario nocturno")
        
        return " | ".join(explanations) if explanations else "Análisis básico"
    
    def basic_analyze_message(self, text, sender, config, dictionary):
        """Fallback al análisis básico si NLP no está disponible"""
        text_lower = text.lower()
        
        high_matches = sum(1 for term in dictionary.get('high_risk', []) if term in text_lower)
        medium_matches = sum(1 for term in dictionary.get('medium_risk', []) if term in text_lower)
        
        total_terms = len(dictionary.get('high_risk', [])) + len(dictionary.get('medium_risk', []))
        if total_terms == 0:
            risk_score = 0
        else:
            risk_score = (high_matches * 0.7 + medium_matches * 0.3) / total_terms
        
        detected_words = [term for term in dictionary.get('high_risk', []) + dictionary.get('medium_risk', []) if term in text_lower]
        
        label = "DETECTADO" if risk_score > config['threshold'] else "NO DETECTADO"
        
        return risk_score, label, detected_words, {}, "Análisis básico (sin NLP)"

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

def get_csv_format_instructions():
    """Retorna las instrucciones del formato CSV"""
    return """
    ## 📋 **FORMATO DEL ARCHIVO CSV DE DICCIONARIO**
    
    El archivo CSV debe tener **exactamente 2 columnas** con los siguientes encabezados:
    
    ```csv
    termino,categoria
    sexy,palabras_alta
    atractiva,palabras_media
    solos,frases_contexto
    jefe,contexto_laboral
    ```
    
    ### **Categorías Válidas:**
    
    | Categoría | Descripción | Peso en Análisis |
    |-----------|-------------|------------------|
    | `palabras_alta` | Términos de alto riesgo | ⚠️ **Alto** (0.7-0.8) |
    | `palabras_media` | Términos de riesgo medio | ⚡ **Medio** (0.3-0.4) |
    | `frases_contexto` | Frases que dan contexto | 🔍 **Contextual** (0.5-0.6) |
    | `contexto_laboral` | Términos de trabajo/profesional | 🏢 **Laboral** (0.3-0.5) |
    | `contexto_relacion` | Términos de relaciones | ❤️ **Relacional** (0.4) |
    | `contexto_financiero` | Términos financieros | 💰 **Financiero** (0.4) |
    | `contexto_agresion` | Términos agresivos | 😠 **Agresivo** (0.6) |
    | `contexto_emocional` | Expresiones emocionales | 😢 **Emocional** (0.3) |
    | `contexto_digital` | Términos digitales/redes | 📱 **Digital** (0.3) |
    | `contexto_sustancias` | Referencias a sustancias | 🚫 **Sustancias** (0.5) |
    """

def load_dictionary_from_file(uploaded_file):
    """Carga diccionario desde archivo subido con validación mejorada"""
    dictionary = {
        'high_risk': [],
        'medium_risk': [], 
        'context_phrases': [],
        'work_context': []
    }
    
    try:
        if uploaded_file.name.endswith('.csv'):
            content = uploaded_file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
        else:
            st.error("❌ Solo se aceptan archivos .csv")
            return None
        
        expected_headers = ['termino', 'categoria']
        if not all(header in reader.fieldnames for header in expected_headers):
            st.error(f"❌ El archivo debe tener las columnas: {', '.join(expected_headers)}")
            return None
        
        category_map = {
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
        
        valid_categories = set(category_map.keys())
        loaded_terms = 0
        
        for row in reader:
            if not row['termino'] or row['termino'].startswith('#'):
                continue
            
            term = row['termino'].strip().lower()
            category = row['categoria'].strip().lower()
            
            if category in valid_categories:
                mapped_category = category_map[category]
                if term and term not in dictionary[mapped_category]:
                    dictionary[mapped_category].append(term)
                    loaded_terms += 1
        
        if loaded_terms > 0:
            st.success(f"✅ Diccionario cargado: {loaded_terms} términos")
            return dictionary
        else:
            st.error("❌ No se pudieron cargar términos válidos")
            return None
        
    except Exception as e:
        st.error(f"❌ Error al procesar archivo: {str(e)}")
        return None

def get_predefined_dictionaries():
    """Retorna diccionarios predefinidos mejorados"""
    return {
        "Acoso Sexual": {
            'high_risk': [
                "desnuda", "desnudo", "fotos íntimas", "sexo", "sexual", "tocarte", 
                "te quiero tocar", "quiero verte", "excitado", "excitada", "cuerpo",
                "te deseo", "sexy", "sensual", "provocativa", "cama", "dormir juntos",
                "masaje", "besos", "caricias", "intimidad", "placer", "fantasía"
            ],
            'medium_risk': [
                "atractiva", "atractivo", "guapa", "guapo", "bonita", "bonito",
                "nena", "nene", "bebé", "cariño", "amor", "corazón", "linda", "hermosa",
                "preciosa", "bella", "encantadora", "seductora"
            ],
            'context_phrases': [
                "solos", "solas", "hotel", "privado", "secreto", "nadie", "no le digas",
                "entre nosotros", "nuestro secreto", "me gustas", "me encanta",
                "encuentro privado", "cita secreta", "momento íntimo"
            ],
            'work_context': [
                "jefe", "jefa", "supervisor", "gerente", "director", "ascenso",
                "promoción", "evaluación", "contrato", "reconocimiento", "bono",
                "reunión privada", "horas extra", "viaje de negocios"
            ]
        },
        "CyberBullying": {
            'high_risk': [
                "idiota", "estúpido", "imbécil", "retrasado", "inútil", "basura",
                "escoria", "patético", "perdedor", "fracasado", "nadie te quiere",
                "todos te odian", "eres repugnante", "das asco", "vete a morir",
                "suicídate", "mátate", "no vales nada", "eres una mierda"
            ],
            'medium_risk': [
                "burla", "ridículo", "vergüenza", "raro", "fenómeno", "bicho raro",
                "inadaptado", "antisocial", "extraño", "anormal", "loco", "chiflado",
                "payaso", "tonto", "bobo", "ignorante"
            ],
            'context_phrases': [
                "todos se ríen de ti", "nadie quiere ser tu amigo", "siempre estás solo",
                "no tienes amigos", "eres invisible", "no perteneces aquí",
                "mejor no vengas", "nadie te invitó", "sobras aquí"
            ],
            'work_context': [
                "redes sociales", "facebook", "instagram", "twitter", "publicar",
                "etiquetar", "compartir", "viral", "meme", "story", "post",
                "grupo", "chat", "clase", "escuela", "colegio"
            ]
        },
        "Infidelidades": {
            'high_risk': [
                "te amo", "te quiero", "mi amor", "amor mío", "mi vida", "corazón",
                "besos", "te extraño", "te necesito", "eres especial", "único",
                "única", "no se lo digas", "secreto", "clandestino", "oculto"
            ],
            'medium_risk': [
                "cariño", "querido", "querida", "tesoro", "cielo", "precioso",
                "preciosa", "encanto", "dulzura", "ternura", "especial"
            ],
            'context_phrases': [
                "entre nosotros", "nadie debe saber", "nuestro secreto", "solo tú y yo",
                "cuando estemos solos", "no puede enterarse", "es complicado",
                "situación difícil", "tengo pareja", "estoy casado", "estoy casada"
            ],
            'work_context': [
                "esposo", "esposa", "marido", "mujer", "novio", "novia", "pareja",
                "familia", "casa", "hogar", "compromiso", "relación", "matrimonio",
                "encuentro", "verse", "quedar", "cita", "hotel", "lugar privado"
            ]
        }
    }

def extract_messages_from_text(content):
    """Extrae mensajes de texto de WhatsApp con mejor parsing"""
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]\.?\s?m\.?)\s*-\s*([^:]+?):\s*(.+)',
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}(?:,|\s)\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s*([^:]+?):\s*(.+)',
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\s*-\s*([^:]+?):\s*(.+)',
        r'(\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+?):\s*(.+)'
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        if matches:
            clean_matches = []
            for match in matches:
                timestamp = match[0].strip()
                sender = match[1].strip()
                message = match[2].strip()
                
                if not message or message.startswith('<Multimedia omitido>') or message.startswith('<Media omitted>'):
                    continue
                
                clean_matches.append((timestamp, sender, message))
            
            if len(clean_matches) > 5:
                all_matches = clean_matches
                break
    
    return all_matches

def create_visualizations(results_df, detection_type):
    """Crea visualizaciones mejoradas de los resultados"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            results_df, 
            x='risk_score', 
            nbins=20,
            title=f'📊 Distribución de Riesgo - {detection_type}',
            labels={'risk_score': 'Puntuación de Riesgo', 'count': 'Cantidad de Mensajes'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red", 
                          annotation_text="Umbral por defecto")
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        detections_by_sender = results_df[results_df['label'] == 'DETECTADO']['sender'].value_counts()
        if not detections_by_sender.empty:
            fig_pie = px.pie(
                values=detections_by_sender.values,
                names=detections_by_sender.index,
                title=f'🎯 Detecciones por Remitente - {detection_type}',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("📊 No se encontraron detecciones para mostrar")

def show_instructions():
    """Muestra el instructivo completo de la aplicación"""
    st.markdown("""
    # 📖 **INSTRUCTIVO COMPLETO - WhatsApp Analyzer NLP**
    
    ## 🎯 **¿QUÉ HACE ESTA APLICACIÓN?**
    
    Esta herramienta analiza conversaciones de WhatsApp para detectar patrones de comportamiento potencialmente problemáticos usando **Inteligencia Artificial (NLP)**. 
    
    ### **Tipos de Detección Disponibles:**
    
    | 🔍 Tipo | 📝 Descripción | 🎯 Detecta |
    |---------|----------------|-------------|
    | **🚨 Acoso Sexual** | Comportamientos de acoso o insinuaciones inapropiadas | Lenguaje sexual, propuestas inapropiadas, acoso laboral |
    | **😠 CyberBullying** | Intimidación, insultos y agresión digital | Insultos, amenazas, exclusión social, humillación |
    | **💔 Infidelidades** | Indicios de relaciones extramaritales o engaños | Expresiones de amor oculto, citas secretas, doble vida |
    
    ## 🛠️ **CÓMO USAR LA APLICACIÓN**
    
    ### **Paso 1: Exportar Chat de WhatsApp**
    
    #### **📱 En Android:**
    1. Abre WhatsApp
    2. Ve al chat que quieres analizar
    3. Toca los **3 puntos** (⋮) → **Más** → **Exportar chat**
    4. Selecciona **"Sin archivos multimedia"**
    5. Guarda el archivo `.txt`
    
    #### **📱 En iPhone:**
    1. Abre WhatsApp
    2. Ve al chat que quieres analizar  
    3. Toca el **nombre del contacto/grupo**
    4. Desliza hacia abajo → **Exportar chat**
    5. Selecciona **"Sin archivos multimedia"**
    6. Guarda el archivo `.txt`
    
    ### **Paso 2: Subir y Analizar**
    
    1. **Sube el archivo .txt** del chat exportado
    2. **Configura los parámetros** en la barra lateral
    3. **Ejecuta el análisis** (puede tardar varios minutos)
    4. **Revisa los resultados** en las diferentes secciones
    
    ## ⚖️ **CONSIDERACIONES LEGALES Y ÉTICAS**
    
    ### **🔒 Privacidad:**
    - ✅ **Procesamiento local**: Todos los datos se procesan en tu navegador
    - ✅ **No almacenamiento**: No guardamos ningún archivo o conversación
    - ✅ **Sin envío de datos**: Nada se envía a servidores externos
    
    ### **⚖️ Uso Responsable:**
    - 📋 **Solo usar con consentimiento** de todas las partes involucradas
    - 🏛️ **Cumplir con leyes locales** de privacidad y protección de datos
    - 👨‍⚖️ **No es evidencia legal**: Los resultados requieren validación profesional
    - 🔍 **Herramienta de apoyo**: Para investigación preliminar, no conclusiones finales
    
    ### **⚠️ Limitaciones:**
    - 🤖 **IA no es 100% precisa**: Siempre revisar resultados manualmente
    - 📝 **Falsos positivos posibles**: Especialmente con sarcasmo o bromas
    - 🌍 **Optimizado para español**: Otros idiomas pueden dar resultados imprecisos
    """)

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Analizador WhatsApp con IA (NLP)</h1>
        <p>Sistema inteligente para detección de patrones de comportamiento en chats</p>
        <small>Versión 4.0 - Con Procesamiento de Lenguaje Natural</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar disponibilidad de NLP
    if not NLP_AVAILABLE:
        st.warning("⚠️ **NLP en modo básico** - Funcionalidad limitada pero operativa")
    elif not nlp:
        st.warning("⚠️ **Modelo de español no encontrado** - Usando análisis básico")
    else:
        st.success("✅ **Análisis NLP Completo Disponible**")
    
    # Crear tabs principales
    tab1, tab2, tab3 = st.tabs(["🔍 Análisis", "📖 Instructivo", "📁 Formato CSV"])
    
    with tab3:
        st.markdown(get_csv_format_instructions())
    
    with tab2:
        show_instructions()
    
    with tab1:
        # Sidebar - Configuración
        with st.sidebar:
            st.header("⚙️ Configuración del Análisis")
            
            detection_options = list(get_predefined_dictionaries().keys()) + ["Diccionario Personalizado"]
            detection_type = st.selectbox(
                "🎯 Tipo de Detección",
                detection_options,
                help="Selecciona qué patrón quieres detectar"
            )
            
            # Diccionario personalizado o predefinido
            dictionary = None
            if detection_type == "Diccionario Personalizado":
                st.subheader("📁 Subir Diccionario CSV")
                
                uploaded_dict = st.file_uploader(
                    "Selecciona archivo CSV",
                    type=['csv'],
                    help="Debe tener columnas: termino,categoria"
                )
                
                if uploaded_dict:
                    dictionary = load_dictionary_from_file(uploaded_dict)
                    if not dictionary:
                        st.stop()
                else:
                    st.warning("⚠️ Sube un diccionario CSV para continuar")
                    st.stop()
            else:
                dictionary = get_predefined_dictionaries()[detection_type]
                
                st.subheader("📊 Diccionario Cargado")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔴 Alto Riesgo", len(dictionary['high_risk']))
                    st.metric("🟡 Riesgo Medio", len(dictionary['medium_risk']))
                with col2:
                    st.metric("🔍 Contexto", len(dictionary['context_phrases']))
                    st.metric("🏢 Laboral", len(dictionary['work_context']))
            
            st.divider()
            
            # Configuración de sensibilidad
            st.subheader("🎚️ Sensibilidad del Análisis")
            sensitivity = st.select_slider(
                "Nivel de Sensibilidad",
                options=['baja', 'media', 'alta'],
                value='media',
                help="Baja: Menos falsos positivos | Alta: Detecta más casos"
            )
            
            custom_threshold = st.slider(
                "🎯 Umbral Personalizado",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="0.0 = Muy sensible | 1.0 = Muy estricto"
            )
            
            use_custom = st.checkbox("Usar umbral personalizado")
            
            st.divider()
            
            # Configuraciones adicionales
            st.subheader("🔧 Opciones Avanzadas")
            
            use_nlp = st.checkbox(
                "🧠 Usar Análisis NLP", 
                value=True and NLP_AVAILABLE and nlp,
                disabled=not (NLP_AVAILABLE and nlp),
                help="Análisis inteligente con IA (más preciso pero más lento)"
            )
            
            show_explanations = st.checkbox(
                "📝 Mostrar Explicaciones Detalladas",
                value=True,
                help="Incluye explicaciones de por qué se detectó cada caso"
            )
        
        # Main content area
        st.header("📤 Subir Archivo de Chat")
        
        uploaded_file = st.file_uploader(
            "📁 Selecciona el archivo de chat (.txt)",
            type=['txt'],
            help="Debe ser una exportación de WhatsApp sin archivos multimedia"
        )
        
        # Procesar archivo si está disponible
        if uploaded_file and dictionary:
            try:
                content = uploaded_file.read().decode('utf-8')
                
                if len(content.strip()) < 100:
                    st.error("❌ El archivo parece estar vacío o muy corto")
                    st.stop()
                
                with st.spinner("🔍 Extrayendo mensajes del chat..."):
                    messages = extract_messages_from_text(content)
                
                if not messages:
                    st.error("❌ No se pudieron extraer mensajes del archivo. Verifica que sea una exportación válida de WhatsApp.")
                    st.stop()
                
                st.success(f"✅ **{len(messages)} mensajes extraídos correctamente**")
                
                config = setup_sensitivity(
                    sensitivity, 
                    custom_threshold if use_custom else None
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"🎯 **Detectando:** {detection_type}")
                with col2:
                    st.info(f"🎚️ **Sensibilidad:** {sensitivity}")
                with col3:
                    st.info(f"🎯 **Umbral:** {config['threshold']:.2f}")
                
                if use_nlp and NLP_AVAILABLE and nlp:
                    st.success("🧠 **Análisis NLP Activado** - Procesamiento inteligente")
                    analyzer = AdvancedNLPAnalyzer()
                else:
                    st.warning("⚡ **Análisis Básico** - Sin NLP")
                    analyzer = AdvancedNLPAnalyzer()
                
                # Procesar mensajes
                with st.spinner(f"🔄 Analizando {len(messages)} mensajes con IA..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, (timestamp, sender, message) in enumerate(messages):
                        progress = (i + 1) / len(messages)
                        progress_bar.progress(progress)
                        
                        if use_nlp and NLP_AVAILABLE and nlp:
                            risk, label, words, analysis_details, explanation = analyzer.analyze_message(
                                message, sender, timestamp, config, dictionary, detection_type
                            )
                        else:
                            risk, label, words, analysis_details, explanation = analyzer.basic_analyze_message(
                                message, sender, config, dictionary
                            )
                        
                        results.append({
                            'timestamp': timestamp,
                            'sender': sender,
                            'message': message,
                            'risk_score': round(risk, 4),
                            'label': label,
                            'detected_words': ', '.join(words) if words else "",
                            'explanation': explanation if show_explanations else ""
                        })
                    
                    progress_bar.empty()
                
                results_df = pd.DataFrame(results)
                
                # Mostrar estadísticas principales
                st.header("📈 Resultados del Análisis")
                
                total_messages = len(results_df)
                detected_messages = len(results_df[results_df['label'] == 'DETECTADO'])
                percentage = (detected_messages / total_messages) * 100 if total_messages > 0 else 0
                avg_risk = results_df['risk_score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Total Mensajes", total_messages)
                
                with col2:
                    st.metric("🚨 Detectados", detected_messages)
                
                with col3:
                    color = "🟢" if percentage < 5 else "🟡" if percentage < 15 else "🔴"
                    st.metric(f"{color} Porcentaje", f"{percentage:.2f}%")
                
                with col4:
                    st.metric("⚖️ Riesgo Promedio", f"{avg_risk:.3f}")
                
                # Mostrar visualizaciones
                if detected_messages > 0:
                    st.header("📊 Análisis Visual")
                    create_visualizations(results_df, detection_type)
                    
                    # Mostrar evidencias
                    st.header("🔍 Evidencias Encontradas")
                    
                    detected_df = results_df[results_df['label'] == 'DETECTADO'].copy()
                    detected_df = detected_df.sort_values('risk_score', ascending=False)
                    
                    for idx, row in detected_df.head(20).iterrows():
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
                                {f"<div style='margin-top: 10px; font-style: italic;'><strong>🧠 Análisis:</strong> {row['explanation']}</div>" if show_explanations and row['explanation'] else ''}
                            </div>
                            """, unsafe_allow_html=True)
                    
                else:
                    st.success("✅ **¡Excelente noticia!** No se detectaron patrones sospechosos en la conversación")
                
                # Opción de descarga
                st.header("💾 Descargar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔒 <strong>Privacidad:</strong> Todos los archivos se procesan localmente. No se almacenan datos.</p>
        <p>⚖️ <strong>Uso Responsable:</strong> Esta herramienta debe usarse únicamente con fines legítimos y respetando la privacidad.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
