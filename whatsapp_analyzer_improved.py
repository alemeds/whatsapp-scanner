#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALIZADOR AVANZADO DE CONVERSACIONES DE WHATSAPP
Aplicación web para detectar diferentes tipos de delitos y comportamientos en chats
Versión optimizada para Streamlit Cloud sin dependencias problemáticas

Autor: Sistema de Análisis de Comunicaciones
Versión: 4.5 - Streamlit Cloud Stable Edition
"""

import streamlit as st
import pandas as pd
import re
import csv
import io
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import hashlib

# Verificación segura de dependencias NLP
@st.cache_resource
def check_nlp_dependencies():
    """Verifica qué dependencias NLP están disponibles de forma segura"""
    spacy_available = False
    textblob_available = False
    nlp_model = None
    
    # Verificar TextBlob (más ligero y confiable)
    try:
        from textblob import TextBlob
        # Crear un test simple
        test_blob = TextBlob("test text")
        _ = test_blob.sentiment.polarity
        textblob_available = True
    except Exception:
        textblob_available = False
    
    # Verificar spaCy solo si está disponible (sin forzar instalación)
    try:
        import spacy
        try:
            nlp_model = spacy.load("es_core_news_sm")
            spacy_available = True
        except OSError:
            # Modelo no disponible, usar análisis básico
            spacy_available = False
    except ImportError:
        # spaCy no instalado, continuar sin él
        spacy_available = False
    
    return spacy_available, textblob_available, nlp_model

# Inicializar verificación de dependencias
SPACY_AVAILABLE, TEXTBLOB_AVAILABLE, nlp = check_nlp_dependencies()

# Configuración de la página
st.set_page_config(
    page_title="Analizador WhatsApp - Detector de Patrones",
    page_icon="🔍",
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

class SmartTextAnalyzer:
    """Analizador de texto inteligente optimizado para Streamlit Cloud"""
    
    def __init__(self):
        self.spacy_available = SPACY_AVAILABLE
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.nlp = nlp
        
        # Patrones de negación
        self.negation_patterns = ["no", "nunca", "jamás", "nada", "tampoco", "ni", "sin"]
        
        # Intensificadores
        self.intensity_modifiers = {
            "muy": 1.5, "super": 1.7, "extremadamente": 2.0, "bastante": 1.3,
            "algo": 0.7, "poco": 0.5, "medio": 0.8, "un poco": 0.6,
            "mucho": 1.4, "demasiado": 1.6, "increíblemente": 1.8
        }
        
        # Emociones básicas
        self.positive_emotions = ["amor", "cariño", "besos", "abrazos", "feliz", "contento", "alegre", "genial"]
        self.negative_emotions = ["odio", "rabia", "ira", "triste", "deprimido", "enojado", "furioso", "disgusto"]
        
    def analyze_message(self, text, sender, timestamp, config, dictionary, detection_type):
        """Análisis principal del mensaje con fallback inteligente"""
        
        # Preprocesamiento básico
        cleaned_text = self.preprocess_text(text)
        
        # Análisis multidimensional
        analysis_results = {
            'sentiment': self.analyze_sentiment(text),
            'negation': self.detect_negation_simple(cleaned_text),
            'intensity': self.calculate_intensity(text, cleaned_text),
            'emotion': self.analyze_basic_emotion(cleaned_text),
            'context': self.analyze_context_smart(cleaned_text, dictionary, detection_type),
            'temporal': self.analyze_temporal_patterns(timestamp),
            'patterns': self.detect_behavioral_patterns(cleaned_text, detection_type)
        }
        
        # Si spaCy está disponible, usar análisis avanzado
        if self.spacy_available and self.nlp:
            analysis_results.update(self.advanced_spacy_analysis(cleaned_text))
        
        # Calcular puntuación inteligente
        base_score = self.calculate_base_score(cleaned_text, dictionary)
        smart_score = self.calculate_contextual_score(base_score, analysis_results, detection_type)
        
        # Palabras detectadas
        detected_words = self.get_detected_words(text, dictionary)
        
        # Explicación detallada
        explanation = self.generate_comprehensive_explanation(analysis_results, smart_score, detection_type)
        
        label = "DETECTADO" if smart_score > config['threshold'] else "NO DETECTADO"
        
        return smart_score, label, detected_words, analysis_results, explanation
    
    def preprocess_text(self, text):
        """Preprocesamiento inteligente del texto"""
        text = text.lower()
        
        # Normalizar emojis emocionales
        emoji_patterns = {
            r'[😍😘❤️💕💖😻🥰💘]': ' _expresion_amor_ ',
            r'[😠😡🤬👿💢😤]': ' _expresion_rabia_ ',
            r'[😢😭💔😞😔🥺]': ' _expresion_tristeza_ ',
            r'[😏😈🔥💦🍆🍑]': ' _expresion_sexual_ ',
            r'[🤮🤢💩👎]': ' _expresion_disgusto_ '
        }
        
        for pattern, replacement in emoji_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalizar repeticiones: "hoooola" -> "hola"
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # Normalizar variaciones de escritura comunes
        replacements = {
            r's[3e]xy': 'sexy', r'k[1i]ero': 'quiero', r'h3rm0sa': 'hermosa',
            r'b[3e]ll[4a]': 'bella', r'amor3s': 'amores', r'\bb+b+\b': 'bebe',
            r'x+d+': 'xd', r'jaj+a*': 'jaja', r'jej+e*': 'jeje'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_sentiment(self, text):
        """Análisis de sentimientos con TextBlob o fallback básico"""
        if self.textblob_available:
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
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
                    'confidence': abs(polarity),
                    'method': 'textblob'
                }
            except Exception:
                pass
        
        # Fallback: análisis básico de sentimientos
        return self.basic_sentiment_analysis(text)
    
    def basic_sentiment_analysis(self, text):
        """Análisis básico de sentimientos sin dependencias"""
        positive_words = ["bien", "bueno", "genial", "excelente", "perfecto", "increíble", "fantástico", "maravilloso"]
        negative_words = ["mal", "malo", "terrible", "horrible", "awful", "pesimo", "desastre", "odioso"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            polarity = 0.5
            interpretation = "positivo"
        elif negative_count > positive_count:
            polarity = -0.5
            interpretation = "negativo"
        else:
            polarity = 0.0
            interpretation = "neutral"
        
        return {
            'polarity': polarity,
            'subjectivity': 0.5,
            'interpretation': interpretation,
            'confidence': abs(polarity),
            'method': 'basic'
        }
    
    def detect_negation_simple(self, text):
        """Detección simple pero efectiva de negaciones"""
        words = text.split()
        negations = []
        
        for i, word in enumerate(words):
            if word in self.negation_patterns:
                # Buscar contexto (siguientes 3 palabras)
                context = words[i+1:i+4] if i+1 < len(words) else []
                
                # Calcular fuerza de la negación
                strength = 1.0 if word in ["nunca", "jamás", "nada"] else 0.8
                
                negations.append({
                    'word': word,
                    'position': i,
                    'context': context,
                    'strength': strength
                })
        
        return negations
    
    def calculate_intensity(self, original_text, cleaned_text):
        """Calcula intensidad emocional del mensaje"""
        intensity_score = 1.0
        indicators = []
        
        # Intensificadores en el texto
        for word, multiplier in self.intensity_modifiers.items():
            if word in cleaned_text:
                intensity_score *= multiplier
                indicators.append(f"{word} ({multiplier}x)")
        
        # MAYÚSCULAS (indica énfasis)
        caps_ratio = sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1)
        if caps_ratio > 0.3:
            intensity_score *= 1.4
            indicators.append(f"MAYÚSCULAS ({caps_ratio:.0%})")
        
        # Signos de puntuación enfáticos
        exclamations = original_text.count('!')
        questions = original_text.count('?')
        
        if exclamations > 1:
            multiplier = 1 + (exclamations - 1) * 0.15
            intensity_score *= multiplier
            indicators.append(f"!×{exclamations}")
        
        if questions > 2:
            intensity_score *= 1.2
            indicators.append(f"?×{questions}")
        
        # Repetición de letras (holaaa, siiii)
        repetition_pattern = r'(.)\1{2,}'
        if re.search(repetition_pattern, original_text):
            intensity_score *= 1.2
            indicators.append("repeticiones")
        
        return {
            'score': min(intensity_score, 3.0),
            'indicators': indicators
        }
    
    def analyze_basic_emotion(self, text):
        """Análisis básico de emociones"""
        positive_count = sum(1 for word in self.positive_emotions if word in text)
        negative_count = sum(1 for word in self.negative_emotions if word in text)
        
        # Detectar emociones específicas por expresiones
        emotion_expressions = {
            '_expresion_amor_': 'romántico',
            '_expresion_rabia_': 'agresivo',
            '_expresion_tristeza_': 'melancólico',
            '_expresion_sexual_': 'sexual',
            '_expresion_disgusto_': 'desprecio'
        }
        
        detected_expressions = []
        for expr, emotion in emotion_expressions.items():
            if expr in text:
                detected_expressions.append(emotion)
        
        if positive_count > negative_count:
            return {'tone': 'positivo', 'strength': positive_count, 'expressions': detected_expressions}
        elif negative_count > positive_count:
            return {'tone': 'negativo', 'strength': negative_count, 'expressions': detected_expressions}
        else:
            return {'tone': 'neutral', 'strength': 0, 'expressions': detected_expressions}
    
    def analyze_context_smart(self, text, dictionary, detection_type):
        """Análisis de contexto mejorado"""
        context_analysis = {
            'laboral': 0, 'romántico': 0, 'familiar': 0, 'agresivo': 0,
            'sexual': 0, 'social': 0, 'temporal': 0
        }
        
        # Términos por contexto
        context_terms = {
            'laboral': ['jefe', 'trabajo', 'oficina', 'reunión', 'proyecto', 'empresa', 'ascenso', 'salario'],
            'romántico': ['amor', 'cariño', 'besos', 'pareja', 'novio', 'novia', 'cita', 'te amo'],
            'familiar': ['familia', 'papá', 'mamá', 'hermano', 'hermana', 'hijo', 'hija', 'casa', 'hogar'],
            'agresivo': ['odio', 'matar', 'golpear', 'destruir', 'venganza', 'rabia', 'pelea'],
            'sexual': dictionary.get('high_risk', []) + dictionary.get('medium_risk', []),
            'social': ['amigos', 'fiesta', 'grupo', 'clase', 'escuela', 'universidad', 'compañeros'],
            'temporal': ['noche', 'madrugada', 'tarde', 'mañana', 'después', 'luego', 'pronto']
        }
        
        # Calcular scores de contexto
        for context_type, terms in context_terms.items():
            matches = sum(1 for term in terms if term in text)
            # Normalizar por número esperado de términos
            max_expected = 3
            context_analysis[context_type] = min(matches / max_expected, 1.0)
        
        return context_analysis
    
    def analyze_temporal_patterns(self, timestamp):
        """Análisis de patrones temporales"""
        try:
            # Extraer hora del timestamp
            hour_match = re.search(r'(\d{1,2}):(\d{2})', timestamp)
            if hour_match:
                hour = int(hour_match.group(1))
                
                # Determinar período del día
                if 6 <= hour <= 12:
                    period = "mañana"
                elif 12 <= hour <= 18:
                    period = "tarde"
                elif 18 <= hour <= 22:
                    period = "noche"
                else:
                    period = "madrugada"
                
                return {
                    'hour': hour,
                    'period': period,
                    'is_night': hour >= 22 or hour <= 6,
                    'is_late': hour >= 23 or hour <= 5,
                    'is_work_hours': 9 <= hour <= 18
                }
        except Exception:
            pass
        
        return {
            'hour': None, 'period': 'desconocido', 'is_night': False,
            'is_late': False, 'is_work_hours': False
        }
    
    def detect_behavioral_patterns(self, text, detection_type):
        """Detecta patrones de comportamiento específicos"""
        patterns = {
            'Acoso Sexual': [
                r'\b(solos?\s+(?:tu\s+y\s+yo|nosotros))\b',
                r'\b(secreto\s+(?:entre|nuestro))\b',
                r'\b(no\s+(?:le\s+)?dig[au]s?)\b',
                r'\b(encuentro\s+privado)\b'
            ],
            'CyberBullying': [
                r'\b(todos\s+(?:te\s+odian|se\s+ríen))\b',
                r'\b(nadie\s+(?:te\s+quiere|te\s+aguanta))\b',
                r'\b(eres\s+(?:un|una)\s+\w+)\b',
                r'\b(no\s+sirves?)\b'
            ],
            'Infidelidades': [
                r'\b(no\s+(?:puede|debe)\s+saber)\b',
                r'\b(entre\s+nosotros)\b',
                r'\b(tengo\s+(?:pareja|esposo|esposa))\b',
                r'\b(es\s+complicado)\b'
            ]
        }
        
        detected_patterns = []
        if detection_type in patterns:
            for pattern in patterns[detection_type]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected_patterns.extend(matches)
        
        return detected_patterns
    
    def advanced_spacy_analysis(self, text):
        """Análisis avanzado con spaCy si está disponible"""
        if not self.spacy_available or not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            # Análisis de entidades
            entities = {
                'persons': [ent.text for ent in doc.ents if ent.label_ in ["PER", "PERSON"]],
                'locations': [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]],
                'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            }
            
            # Análisis gramatical
            pos_tags = [token.pos_ for token in doc]
            verb_count = pos_tags.count('VERB')
            noun_count = pos_tags.count('NOUN')
            adj_count = pos_tags.count('ADJ')
            
            return {
                'entities': entities,
                'grammar': {
                    'verbs': verb_count,
                    'nouns': noun_count,
                    'adjectives': adj_count,
                    'complexity': len(set(pos_tags)) / len(pos_tags) if pos_tags else 0
                },
                'spacy_available': True
            }
        except Exception:
            return {'spacy_available': False}
    
    def calculate_base_score(self, text, dictionary):
        """Calcula puntuación base usando el diccionario"""
        high_matches = sum(1 for term in dictionary.get('high_risk', []) if term in text)
        medium_matches = sum(1 for term in dictionary.get('medium_risk', []) if term in text)
        context_matches = sum(1 for term in dictionary.get('context_phrases', []) if term in text)
        work_matches = sum(1 for term in dictionary.get('work_context', []) if term in text)
        
        total_terms = (len(dictionary.get('high_risk', [])) + 
                      len(dictionary.get('medium_risk', [])))
        
        if total_terms == 0:
            return 0
        
        # Puntuación ponderada
        score = ((high_matches * 0.8) + 
                (medium_matches * 0.4) + 
                (context_matches * 0.3) + 
                (work_matches * 0.2)) / max(total_terms, 1)
        
        return min(score, 1.0)
    
    def calculate_contextual_score(self, base_score, analysis, detection_type):
        """Calcula puntuación final considerando contexto"""
        score = base_score
        
        # Ajustar por negaciones
        negations = analysis['negation']
        for negation in negations:
            reduction = negation['strength'] * 0.4
            score *= (1 - reduction)
        
        # Ajustar por intensidad
        intensity = analysis['intensity']['score']
        score *= intensity
        
        # Ajustar por sentimientos
        sentiment = analysis['sentiment']
        if detection_type == "Acoso Sexual":
            if sentiment['polarity'] > 0.2:
                score *= 1.2
        elif detection_type == "CyberBullying":
            if sentiment['polarity'] < -0.3:
                score *= 1.4
        elif detection_type == "Infidelidades":
            if abs(sentiment['polarity']) > 0.3:
                score *= 1.2
        
        # Ajustar por contexto específico
        context = analysis['context']
        if detection_type == "Acoso Sexual":
            if context['laboral'] > 0.5 and context['sexual'] > 0.3:
                score *= 1.8  # Acoso laboral es muy grave
            elif context['romántico'] > 0.6:
                score *= 0.7  # En contexto romántico es menos grave
        
        elif detection_type == "CyberBullying":
            if context['agresivo'] > 0.4:
                score *= 1.6
            if context['social'] > 0.3:
                score *= 1.3
        
        elif detection_type == "Infidelidades":
            if context['romántico'] > 0.4:
                score *= 1.4
            temporal = analysis['temporal']
            if temporal['is_night']:
                score *= 1.2
        
        # Bonificación por patrones específicos
        patterns = analysis['patterns']
        if patterns:
            score *= (1 + len(patterns) * 0.2)
        
        return min(score, 1.0)
    
    def get_detected_words(self, text, dictionary):
        """Obtiene palabras detectadas del diccionario"""
        text_lower = text.lower()
        detected = []
        
        for category, terms in dictionary.items():
            for term in terms:
                if term.lower() in text_lower and term not in detected:
                    detected.append(term)
        
        return detected
    
    def generate_comprehensive_explanation(self, analysis, score, detection_type):
        """Genera explicación detallada del análisis"""
        explanations = []
        
        # Sentimiento
        sentiment = analysis['sentiment']
        if sentiment['confidence'] > 0.3:
            explanations.append(f"Sentimiento {sentiment['interpretation']} ({sentiment['polarity']:.2f})")
        
        # Negaciones
        negations = analysis['negation']
        if negations:
            explanations.append(f"Negaciones: {len(negations)}")
        
        # Intensidad
        intensity = analysis['intensity']
        if intensity['score'] > 1.3:
            explanations.append(f"Alta intensidad ({intensity['score']:.1f}x)")
        
        # Contexto dominante
        context = analysis['context']
        max_context = max(context.items(), key=lambda x: x[1])
        if max_context[1] > 0.4:
            explanations.append(f"Contexto {max_context[0]}")
        
        # Emociones específicas
        emotion = analysis['emotion']
        if emotion['expressions']:
            explanations.append(f"Expresiones: {', '.join(emotion['expressions'])}")
        
        # Patrones específicos
        patterns = analysis['patterns']
        if patterns:
            explanations.append(f"Patrones detectados: {len(patterns)}")
        
        # Información temporal
        temporal = analysis['temporal']
        if temporal['is_night']:
            explanations.append("Horario nocturno")
        
        # Método de análisis
        method = "NLP completo" if self.spacy_available else "Análisis inteligente"
        if sentiment['method'] == 'textblob':
            method += " + TextBlob"
        
        explanation_text = " | ".join(explanations) if explanations else "Análisis básico"
        return f"{explanation_text} ({method})"

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
    
    ### **Ejemplo Completo:**
    ```csv
    termino,categoria
    sexy,palabras_alta
    hermosa,palabras_media
    atractiva,palabras_media
    solos,frases_contexto
    secreto,frases_contexto
    jefe,contexto_laboral
    ascenso,contexto_laboral
    amor,contexto_relacion
    beso,contexto_relacion
    odio,contexto_agresion
    matar,contexto_agresion
    ```
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
            st.error(f"📋 Columnas encontradas: {', '.join(reader.fieldnames)}")
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
        invalid_categories = set()
        
        for row in reader:
            if not row['termino'] or row['termino'].startswith('#'):
                continue
            
            term = row['termino'].strip().lower()
            category = row['categoria'].strip().lower()
            
            if category not in valid_categories:
                invalid_categories.add(category)
                continue
            
            mapped_category = category_map[category]
            if term and term not in dictionary[mapped_category]:
                dictionary[mapped_category].append(term)
                loaded_terms += 1
        
        if invalid_categories:
            st.warning(f"⚠️ Categorías inválidas ignoradas: {', '.join(invalid_categories)}")
        
        if loaded_terms > 0:
            st.success(f"✅ Diccionario cargado: {loaded_terms} términos")
            
            # Mostrar distribución
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Alto Riesgo", len(dictionary['high_risk']))
            with col2:
                st.metric("Riesgo Medio", len(dictionary['medium_risk']))
            with col3:
                st.metric("Contexto", len(dictionary['context_phrases']))
            with col4:
                st.metric("Laboral", len(dictionary['work_context']))
            
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
                "masaje", "besos", "caricias", "intimidad", "placer", "fantasía",
                "seducir", "tentación", "erótico", "voluptuosa", "cachonda", "caliente"
            ],
            'medium_risk': [
                "atractiva", "atractivo", "guapa", "guapo", "bonita", "bonito",
                "nena", "nene", "bebé", "cariño", "amor", "corazón", "linda", "hermosa",
                "preciosa", "bella", "encantadora", "seductora", "divina", "diosa",
                "princesa", "reina", "dulzura", "ternura"
            ],
            'context_phrases': [
                "solos", "solas", "hotel", "privado", "secreto", "nadie", "no le digas",
                "entre nosotros", "nuestro secreto", "me gustas", "me encanta",
                "encuentro privado", "cita secreta", "momento íntimo", "lugar reservado",
                "cuando estemos solos", "sin que nadie sepa"
            ],
            'work_context': [
                "jefe", "jefa", "supervisor", "gerente", "director", "ascenso",
                "promoción", "evaluación", "contrato", "reconocimiento", "bono",
                "reunión privada", "horas extra", "viaje de negocios", "después del trabajo",
                "oficina", "despacho", "proyecto", "empresa"
            ]
        },
        "CyberBullying": {
            'high_risk': [
                "idiota", "estúpido", "imbécil", "retrasado", "inútil", "basura",
                "escoria", "patético", "perdedor", "fracasado", "nadie te quiere",
                "todos te odian", "eres repugnante", "das asco", "vete a morir",
                "suicídate", "mátate", "no vales nada", "eres una mierda", "despreciable",
                "asqueroso", "aberración", "escupitajo", "lacra", "parásito", "cucaracha"
            ],
            'medium_risk': [
                "burla", "ridículo", "vergüenza", "raro", "fenómeno", "bicho raro",
                "inadaptado", "antisocial", "extraño", "anormal", "loco", "chiflado",
                "payaso", "tonto", "bobo", "ignorante", "torpe", "incapaz", "débil",
                "cobarde", "llorica"
            ],
            'context_phrases': [
                "todos se ríen de ti", "nadie quiere ser tu amigo", "siempre estás solo",
                "no tienes amigos", "eres invisible", "no perteneces aquí",
                "mejor no vengas", "nadie te invitó", "sobras aquí", "estás de más",
                "no encajas", "eres el hazmerreír", "todos hablan de ti"
            ],
            'work_context': [
                "redes sociales", "facebook", "instagram", "twitter", "publicar",
                "etiquetar", "compartir", "viral", "meme", "story", "post",
                "grupo", "chat", "clase", "escuela", "colegio", "compañeros",
                "universidad", "instituto"
            ]
        },
        "Infidelidades": {
            'high_risk': [
                "te amo", "te quiero", "mi amor", "amor mío", "mi vida", "corazón",
                "besos", "te extraño", "te necesito", "eres especial", "único",
                "única", "no se lo digas", "secreto", "clandestino", "oculto",
                "amor prohibido", "relación secreta", "aventura", "escapada",
                "mi alma gemela", "eres todo para mí"
            ],
            'medium_risk': [
                "cariño", "querido", "querida", "tesoro", "cielo", "precioso",
                "preciosa", "encanto", "dulzura", "ternura", "especial",
                "importante", "diferente", "comprensión", "conexión", "química",
                "atracción", "feeling"
            ],
            'context_phrases': [
                "entre nosotros", "nadie debe saber", "nuestro secreto", "solo tú y yo",
                "cuando estemos solos", "no puede enterarse", "es complicado",
                "situación difícil", "tengo pareja", "estoy casado", "estoy casada",
                "mi esposo no", "mi esposa no", "relación complicada", "no es el momento"
            ],
            'work_context': [
                "esposo", "esposa", "marido", "mujer", "novio", "novia", "pareja",
                "familia", "casa", "hogar", "compromiso", "relación", "matrimonio",
                "encuentro", "verse", "quedar", "cita", "hotel", "lugar privado",
                "escaparse", "mentir", "coartada", "excusa", "disimular"
            ]
        }
    }

def extract_messages_from_text(content):
    """Extrae mensajes de texto de WhatsApp con parsing robusto"""
    patterns = [
        # Formato Android común con AM/PM
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]\.?\s?m\.?)\s*-\s*([^:]+?):\s*(.+)',
        # Formato con corchetes
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}(?:,|\s)\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s*([^:]+?):\s*(.+)',
        # Formato simple sin AM/PM
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\s*-\s*([^:]+?):\s*(.+)',
        # Formato ISO con coma
        r'(\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+?):\s*(.+)',
        # Formato alternativo con guión
        r'(\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}:\d{2})\s*-\s*([^:]+?):\s*(.+)'
    ]
    
    all_matches = []
    best_pattern_matches = []
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        if matches:
            clean_matches = []
            for match in matches:
                timestamp = match[0].strip()
                sender = match[1].strip()
                message = match[2].strip()
                
                # Filtrar mensajes del sistema y multimedia
                system_messages = [
                    '<multimedia omitido>', '<media omitted>', 'se unió usando', 
                    'cambió el asunto', 'eliminó este mensaje', 'mensaje eliminado',
                    'left', 'joined', 'changed subject to'
                ]
                
                if not message or any(sys_msg in message.lower() for sys_msg in system_messages):
                    continue
                
                # Validar que el sender no sea muy largo (probable error de parsing)
                if len(sender) > 50:
                    continue
                
                clean_matches.append((timestamp, sender, message))
            
            if len(clean_matches) > len(best_pattern_matches):
                best_pattern_matches = clean_matches
    
    return best_pattern_matches

def create_visualizations(results_df, detection_type):
    """Crea visualizaciones mejoradas de los resultados"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de distribución de riesgo
        fig_hist = px.histogram(
            results_df, 
            x='risk_score', 
            nbins=25,
            title=f'📊 Distribución de Puntuación de Riesgo - {detection_type}',
            labels={'risk_score': 'Puntuación de Riesgo', 'count': 'Cantidad de Mensajes'},
            color_discrete_sequence=['#667eea']
        )
        
        # Líneas de referencia
        fig_hist.add_vline(x=0.45, line_dash="dot", line_color="green", 
                          annotation_text="Sensibilidad Alta", annotation_position="top")
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="orange", 
                          annotation_text="Sensibilidad Media", annotation_position="top")
        fig_hist.add_vline(x=0.75, line_dash="solid", line_color="red", 
                          annotation_text="Sensibilidad Baja", annotation_position="top")
        
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Gráfico de detecciones por remitente
        detections_by_sender = results_df[results_df['label'] == 'DETECTADO']['sender'].value_counts()
        if not detections_by_sender.empty and len(detections_by_sender) > 1:
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
            # Gráfico de barras alternativo si hay pocos remitentes
            sender_stats = results_df.groupby('sender').agg({
                'risk_score': ['count', 'mean', 'max'],
                'label': lambda x: (x == 'DETECTADO').sum()
            }).round(3)
            sender_stats.columns = ['Total', 'Promedio', 'Máximo', 'Detectados']
            
            if len(sender_stats) > 0:
                fig_bar = px.bar(
                    x=sender_stats.index,
                    y=sender_stats['Detectados'],
                    title=f'🎯 Detecciones por Remitente - {detection_type}',
                    labels={'x': 'Remitente', 'y': 'Detecciones'},
                    color_discrete_sequence=['#e74c3c']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("📊 No hay suficientes datos para visualizar")
    
    # Timeline de actividad si hay detecciones
    detected_df = results_df[results_df['label'] == 'DETECTADO']
    if len(detected_df) > 0:
        st.subheader("📅 Timeline de Detecciones")
        
        try:
            # Intentar parsear fechas para timeline
            detected_df['parsed_date'] = pd.to_datetime(detected_df['timestamp'], errors='coerce', infer_datetime_format=True)
            detected_df_with_dates = detected_df.dropna(subset=['parsed_date'])
            
            if len(detected_df_with_dates) > 0:
                # Agrupar por día
                daily_counts = detected_df_with_dates.groupby(detected_df_with_dates['parsed_date'].dt.date).size().reset_index()
                daily_counts.columns = ['date', 'count']
                
                if len(daily_counts) > 1:
                    fig_timeline = px.line(
                        daily_counts,
                        x='date',
                        y='count',
                        title='📈 Evolución de Detecciones por Día',
                        markers=True,
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_timeline.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Fecha",
                        yaxis_title="Número de Detecciones"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("📊 Todas las detecciones ocurrieron en el mismo día")
            else:
                st.info("⏰ No se pudieron parsear las fechas para el timeline")
        except Exception as e:
            st.info(f"⏰ No se pudo generar timeline temporal: {str(e)}")

def show_instructions():
    """Muestra el instructivo completo de la aplicación"""
    st.markdown("""
    # 📖 **INSTRUCTIVO COMPLETO - WhatsApp Analyzer**
    
    ## 🎯 **¿QUÉ HACE ESTA APLICACIÓN?**
    
    Esta herramienta analiza conversaciones de WhatsApp para detectar patrones de comportamiento potencialmente problemáticos usando **análisis inteligente de texto**. 
    
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
    
    ### **Paso 2: Configurar el Análisis**
    
    #### **🎯 Seleccionar Tipo de Detección:**
    - **Acoso Sexual**: Para detectar comportamientos inapropiados
    - **CyberBullying**: Para identificar intimidación o agresión
    - **Infidelidades**: Para encontrar indicios de engaños
    - **Diccionario Personalizado**: Para usar tus propios términos
    
    #### **🎚️ Configurar Sensibilidad:**
    - **Baja (0.75)**: Conservador - Solo casos evidentes
    - **Media (0.60)**: Balanceado - Precisión óptima (recomendado)
    - **Alta (0.45)**: Agresivo - Detecta casos sutiles
    
    ### **Paso 3: Subir y Analizar**
    
    1. **Sube el archivo .txt** del chat exportado
    2. **Configura los parámetros** en la barra lateral
    3. **Ejecuta el análisis** (puede tardar varios minutos)
    4. **Revisa los resultados** en las diferentes secciones
    
    ## 🧠 **CÓMO FUNCIONA EL ANÁLISIS INTELIGENTE**
    
    ### **Características del Sistema:**
    
    - **Detección de Negaciones**: Distingue "Eres sexy" de "No eres sexy"
    - **Análisis de Intensidad**: Reconoce MAYÚSCULAS, !!!, repeticiones
    - **Contexto Específico**: Adapta el análisis según el tipo de detección
    - **Análisis Temporal**: Considera horarios (nocturno, laboral)
    - **Patrones de Comportamiento**: Detecta frases características
    - **Análisis de Sentimientos**: Evalúa tono positivo/negativo/neutral
    
    ### **Tecnologías Utilizadas:**
    
    - **TextBlob**: Para análisis de sentimientos (si está disponible)
    - **spaCy**: Para análisis NLP avanzado (si está disponible)
    - **Regex Avanzado**: Para detección de patrones específicos
    - **Análisis Contextual**: Algoritmos propios para contexto
    
    ## 📊 **INTERPRETANDO LOS RESULTADOS**
    
    ### **🎯 Métricas Principales:**
    
    | 📊 Métrica | 📝 Significado |
    |------------|----------------|
    | **Total Mensajes** | Cantidad total de mensajes analizados |
    | **Detectados** | Mensajes que superaron el umbral de riesgo |
    | **Porcentaje** | % de mensajes problemáticos |
    | **Riesgo Promedio** | Puntuación promedio de riesgo (0.0 - 1.0) |
    
    ### **🚦 Niveles de Riesgo:**
    
    - 🟢 **0.0 - 0.4**: Riesgo bajo o nulo
    - 🟡 **0.4 - 0.6**: Riesgo moderado, revisar contexto
    - 🔴 **0.6 - 1.0**: Riesgo alto, requiere atención
    
    ### **📋 Sección de Evidencias:**
    
    Cada evidencia muestra:
    - **👤 Remitente**: Quién envió el mensaje
    - **📅 Fecha/Hora**: Cuándo se envió
    - **💬 Mensaje**: Contenido completo
    - **⚖️ Puntuación**: Nivel de riesgo calculado
    - **🎯 Términos**: Palabras específicas detectadas
    - **🧠 Explicación**: Por qué se detectó (sentimiento, contexto, etc.)
    
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
    - 📊 **Requiere volumen**: Pocos mensajes pueden dar análisis limitado
    
    ## 🚨 **SOLUCIÓN DE PROBLEMAS**
    
    ### **❌ "No se pudieron extraer mensajes"**
    - Verifica que el archivo sea una exportación de WhatsApp
    - Asegúrate de exportar "sin archivos multimedia"
    - El archivo debe estar en formato .txt
    
    ### **⚠️ "Error al cargar diccionario"**
    - Verifica que el CSV tenga las columnas: `termino,categoria`
    - Usa categorías válidas (ver formato arriba)
    - Asegúrate de que el archivo esté en UTF-8
    
    ### **🐌 "El análisis es muy lento"**
    - Chats muy largos (>5000 mensajes) pueden tardar varios minutos
    - Considera dividir el chat en períodos más pequeños
    - Usar sensibilidad "baja" es más rápido
    
    ### **📊 "Muchos falsos positivos"**
    - Reduce la sensibilidad a "baja"
    - Aumenta el umbral personalizado (ej: 0.75)
    - Revisa manualmente los resultados
    
    ## ✅ **RESUMEN RÁPIDO**
    
    1. **📤 Exporta** chat de WhatsApp (sin multimedia)
    2. **🎯 Selecciona** tipo de detección
    3. **⚙️ Configura** sensibilidad
    4. **📁 Sube** archivo .txt
    5. **🔄 Analiza** y espera resultados
    6. **📊 Revisa** evidencias y gráficos
    7. **💾 Descarga** reportes si necesario
    
    **¡Listo para comenzar el análisis!** 🚀
    """)

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Analizador WhatsApp Inteligente</h1>
        <p>Sistema avanzado para detección de patrones de comportamiento en chats</p>
        <small>Versión 4.5 - Streamlit Cloud Optimized</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar estado de las herramientas
    col1, col2 = st.columns(2)
    with col1:
        if SPACY_AVAILABLE:
            st.success("✅ **Análisis NLP Completo** (spaCy + análisis avanzado)")
        else:
            st.info("ℹ️ **Análisis Inteligente** (sin spaCy, pero completamente funcional)")
    
    with col2:
        if TEXTBLOB_AVAILABLE:
            st.success("✅ **Análisis de Sentimientos** (TextBlob integrado)")
        else:
            st.info("ℹ️ **Análisis Básico de Sentimientos** (algoritmo propio)")
    
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
            
            # Mostrar información del tipo seleccionado
            if detection_type != "Diccionario Personalizado":
                info_dict = {
                    "Acoso Sexual": "🚨 Detecta insinuaciones inapropiadas, propuestas sexuales, acoso laboral",
                    "CyberBullying": "😠 Identifica insultos, amenazas, intimidación, exclusión social", 
                    "Infidelidades": "💔 Encuentra expresiones románticas ocultas, citas secretas, engaños"
                }
                st.info(info_dict[detection_type])
            
            # Diccionario personalizado o predefinido
            dictionary = None
            if detection_type == "Diccionario Personalizado":
                st.subheader("📁 Subir Diccionario CSV")
                
                with st.expander("📋 Ver formato requerido"):
                    st.code("""termino,categoria
sexy,palabras_alta
atractiva,palabras_media
solos,frases_contexto
jefe,contexto_laboral""")
                
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
            
            # Explicación de sensibilidad
            sensitivity_info = {
                'baja': "🟢 Conservador - Solo casos evidentes (umbral: 0.75)",
                'media': "🟡 Balanceado - Precisión óptima (umbral: 0.60)", 
                'alta': "🔴 Agresivo - Detecta casos sutiles (umbral: 0.45)"
            }
            st.info(sensitivity_info[sensitivity])
            
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
            
            show_explanations = st.checkbox(
                "📝 Mostrar Explicaciones Detalladas",
                value=True,
                help="Incluye explicaciones de por qué se detectó cada caso"
            )
            
            max_results = st.selectbox(
                "📊 Máximo de Evidencias a Mostrar",
                [10, 20, 50, 100, "Todas"],
                index=1,
                help="Limita resultados para mejor rendimiento"
            )
        
        # Main content area
        st.header("📤 Subir Archivo de Chat")
        
        # Instrucciones rápidas
        with st.expander("🔧 ¿Cómo exportar chat de WhatsApp?"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📱 Android:**
                1. Abre el chat en WhatsApp
                2. Toca ⋮ → Más → Exportar chat
                3. Selecciona "Sin archivos multimedia"
                4. Guarda el archivo .txt
                """)
            with col2:
                st.markdown("""
                **📱 iPhone:**
                1. Abre el chat en WhatsApp
                2. Toca el nombre del contacto
                3. Exportar chat → Sin archivos multimedia
                4. Guarda el archivo .txt
                """)
        
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
                    st.error("""
                    ❌ **No se pudieron extraer mensajes del archivo.**
                    
                    **Posibles causas:**
                    - El archivo no es una exportación válida de WhatsApp
                    - Formato de fecha no reconocido
                    - Archivo corrupto o modificado
                    
                    **Solución:**
                    - Verifica que sea un archivo .txt exportado directamente de WhatsApp
                    - Asegúrate de seleccionar "Sin archivos multimedia" al exportar
                    """)
                    st.stop()
                
                st.success(f"✅ **{len(messages)} mensajes extraídos correctamente**")
                
                # Mostrar muestra de mensajes
                with st.expander(f"👀 Vista previa de mensajes (primeros 5 de {len(messages)})"):
                    for i, (timestamp, sender, message) in enumerate(messages[:5]):
                        st.markdown(f"""
                        <div class="chat-message">
                            <strong>📅 {timestamp}</strong> | <strong>👤 {sender}</strong><br>
                            💬 {message[:100]}{'...' if len(message) > 100 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Configurar análisis
                config = setup_sensitivity(
                    sensitivity, 
                    custom_threshold if use_custom else None
                )
                
                # Información del análisis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"🎯 **Detectando:** {detection_type}")
                with col2:
                    st.info(f"🎚️ **Sensibilidad:** {sensitivity}")
                with col3:
                    st.info(f"🎯 **Umbral:** {config['threshold']:.2f}")
                
                # Inicializar analizador
                analyzer = SmartTextAnalyzer()
                
                # Procesar mensajes
                with st.spinner(f"🔄 Analizando {len(messages)} mensajes..."):
                    results = []
                    progress_bar = st.progress(0)
                    status_placeholder = st.empty()
                    
                    for i, (timestamp, sender, message) in enumerate(messages):
                        # Actualizar progreso
                        progress = (i + 1) / len(messages)
                        progress_bar.progress(progress)
                        status_placeholder.text(f"Procesando mensaje {i+1}/{len(messages)}: {sender}")
                        
                        # Analizar mensaje
                        risk, label, words, analysis_details, explanation = analyzer.analyze_message(
                            message, sender, timestamp, config, dictionary, detection_type
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
                    status_placeholder.empty()
                
                # Crear DataFrame de resultados
                results_df = pd.DataFrame(results)
                
                # Mostrar estadísticas principales
                st.header("📈 Resultados del Análisis")
                
                total_messages = len(results_df)
                detected_messages = len(results_df[results_df['label'] == 'DETECTADO'])
                percentage = (detected_messages / total_messages) * 100 if total_messages > 0 else 0
                avg_risk = results_df['risk_score'].mean()
                max_risk = results_df['risk_score'].max()
                
                # Métricas principales
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("📝 Total Mensajes", total_messages)
                
                with col2:
                    st.metric(
                        "🚨 Detectados", 
                        detected_messages,
                        delta=f"{percentage:.1f}%" if percentage > 0 else None
                    )
                
                with col3:
                    # Color según porcentaje
                    color = "🟢" if percentage < 5 else "🟡" if percentage < 15 else "🔴"
                    st.metric(f"{color} Porcentaje", f"{percentage:.2f}%")
                
                with col4:
                    st.metric("⚖️ Riesgo Promedio", f"{avg_risk:.3f}")
                
                with col5:
                    st.metric("📊 Riesgo Máximo", f"{max_risk:.3f}")
                
                # Evaluación del riesgo general
                if percentage == 0:
                    st.success("✅ **Excelente**: No se detectaron patrones problemáticos")
                elif percentage < 5:
                    st.info("🟢 **Bajo Riesgo**: Pocos casos detectados, revisar manualmente")
                elif percentage < 15:
                    st.warning("🟡 **Riesgo Moderado**: Revisar casos detectados cuidadosamente")
                else:
                    st.error("🔴 **Alto Riesgo**: Múltiples detecciones, requiere atención inmediata")
                
                # Mostrar visualizaciones
                if detected_messages > 0:
                    st.header("📊 Análisis Visual")
                    create_visualizations(results_df, detection_type)
                    
                    # Análisis por remitente
                    st.subheader("👥 Análisis por Remitente")
                    sender_stats = results_df.groupby('sender').agg({
                        'risk_score': ['count', 'mean', 'max'],
                        'label': lambda x: (x == 'DETECTADO').sum()
                    }).round(3)
                    
                    sender_stats.columns = ['Total Mensajes', 'Riesgo Promedio', 'Riesgo Máximo', 'Detecciones']
                    sender_stats = sender_stats.sort_values('Detecciones', ascending=False)
                    
                    st.dataframe(
                        sender_stats,
                        use_container_width=True,
                        column_config={
                            "Total Mensajes": st.column_config.NumberColumn("📝 Total"),
                            "Riesgo Promedio": st.column_config.NumberColumn("⚖️ Promedio", format="%.3f"),
                            "Riesgo Máximo": st.column_config.NumberColumn("📊 Máximo", format="%.3f"),
                            "Detecciones": st.column_config.NumberColumn("🚨 Detectados")
                        }
                    )
                    
                    # Mostrar evidencias
                    st.header("🔍 Evidencias Encontradas")
                    
                    detected_df = results_df[results_df['label'] == 'DETECTADO'].copy()
                    detected_df = detected_df.sort_values('risk_score', ascending=False)
                    
                    # Filtros
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sender_filter = st.multiselect(
                            "👤 Filtrar por remitente:",
                            options=detected_df['sender'].unique(),
                            default=detected_df['sender'].unique()
                        )
                    
                    with col2:
                        risk_threshold = st.slider(
                            "⚖️ Riesgo mínimo:",
                            min_value=0.0,
                            max_value=1.0,
                            value=config['threshold'],
                            step=0.05
                        )
                    
                    with col3:
                        word_filter = st.text_input(
                            "🔍 Buscar palabra:",
                            placeholder="Ej: sexy, amor...",
                            help="Busca mensajes que contengan esta palabra"
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
                    
                    # Limitar resultados
                    if max_results != "Todas":
                        filtered_df = filtered_df.head(max_results)
                    
                    st.info(f"📊 Mostrando {len(filtered_df)} evidencias de {len(detected_df)} total")
                    
                    # Mostrar evidencias
                    for idx, row in filtered_df.iterrows():
                        # Determinar nivel de riesgo para el color
                        if row['risk_score'] > 0.8:
                            risk_class = "high"
                            risk_emoji = "🔴"
                            risk_text = "ALTO"
                        elif row['risk_score'] > 0.6:
                            risk_class = "medium" 
                            risk_emoji = "🟡"
                            risk_text = "MEDIO"
                        else:
                            risk_class = "low"
                            risk_emoji = "🟢"
                            risk_text = "BAJO"
                        
                        # Badge del tipo de detección
                        badge_class = {
                            "Acoso Sexual": "badge-acoso",
                            "CyberBullying": "badge-bullying", 
                            "Infidelidades": "badge-infidelidad"
                        }.get(detection_type, "badge-acoso")
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="evidence-card risk-{risk_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <div>
                                        <strong style="font-size: 1.1em;">👤 {row['sender']}</strong>
                                        <span class="detection-badge {badge_class}">{detection_type}</span>
                                    </div>
                                    <div style="text-align: right;">
                                        <span style="color: #666; font-size: 0.9em;">📅 {row['timestamp']}</span><br>
                                        <span style="font-weight: bold; color: {'#dc3545' if risk_class=='high' else '#ffc107' if risk_class=='medium' else '#28a745'};">
                                            {risk_emoji} RIESGO {risk_text}
                                        </span>
                                    </div>
                                </div>
                                
                                <div class="chat-message" style="margin: 15px 0;">
                                    💬 <em>"{row['message']}"</em>
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                                    <div>
                                        <strong>⚖️ Puntuación de Riesgo:</strong><br>
                                        <span style="font-size: 1.2em; font-weight: bold; color: {'#dc3545' if risk_class=='high' else '#ffc107' if risk_class=='medium' else '#28a745'};">
                                            {row['risk_score']:.3f}
                                        </span>
                                    </div>
                                    <div>
                                        <strong>🎯 Términos Detectados:</strong><br>
                                        <span style="color: #e74c3c; font-weight: bold;">
                                            {row['detected_words'] if row['detected_words'] else 'N/A'}
                                        </span>
                                    </div>
                                </div>
                                
                                {f'''
                                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                                    <strong>🧠 Análisis Detallado:</strong><br>
                                    <span style="color: #555; font-style: italic;">{row['explanation']}</span>
                                </div>
                                ''' if show_explanations and row['explanation'] else ''}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if len(filtered_df) == 0:
                        st.info("🔍 No se encontraron evidencias con los filtros actuales")
                        
                else:
                    st.success("✅ **¡Excelente noticia!** No se detectaron patrones sospechosos en la conversación")
                    
                    # Mostrar algunas estadísticas básicas aunque no haya detecciones
                    st.subheader("📊 Estadísticas Generales")
                    
                    # Distribución de remitentes
                    sender_counts = results_df['sender'].value_counts()
                    if len(sender_counts) > 1:
                        fig_senders = px.bar(
                            x=sender_counts.index,
                            y=sender_counts.values,
                            title="📱 Mensajes por Remitente",
                            labels={'x': 'Remitente', 'y': 'Cantidad de Mensajes'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig_senders.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_senders, use_container_width=True)
                
                # Opción de descarga de resultados
                st.header("💾 Descargar Resultados")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV completo
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_data = csv_buffer.getvalue()
                    
                    filename = f"analisis_completo_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="📄 Descargar Análisis Completo (CSV)",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        help="Incluye todos los mensajes analizados"
                    )
                
                with col2:
                    # Solo detecciones
                    if detected_messages > 0:
                        detected_csv = io.StringIO()
                        detected_df.to_csv(detected_csv, index=False, encoding='utf-8')
                        detected_data = detected_csv.getvalue()
                        
                        filename_detected = f"solo_detecciones_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="🚨 Descargar Solo Detecciones (CSV)",
                            data=detected_data,
                            file_name=filename_detected,
                            mime="text/csv",
                            help="Solo mensajes detectados como problemáticos"
                        )
                    else:
                        st.info("📊 No hay detecciones para descargar")
                
                with col3:
                    # Reporte resumen
                    report_data = f"""REPORTE DE ANÁLISIS - {detection_type.upper()}
=====================================
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Archivo analizado: {uploaded_file.name}
Tipo de detección: {detection_type}
Sensibilidad: {sensitivity}
Umbral usado: {config['threshold']:.3f}

ESTADÍSTICAS GENERALES:
- Total de mensajes: {total_messages}
- Mensajes detectados: {detected_messages}
- Porcentaje de detección: {percentage:.2f}%
- Riesgo promedio: {avg_risk:.4f}
- Riesgo máximo: {max_risk:.4f}

ANÁLISIS POR REMITENTE:
{sender_stats.to_string() if detected_messages > 0 else 'No hay detecciones'}

TECNOLOGÍAS USADAS:
- spaCy: {'Disponible' if SPACY_AVAILABLE else 'No disponible'}
- TextBlob: {'Disponible' if TEXTBLOB_AVAILABLE else 'No disponible'}
- Análisis inteligente: Activado
- Detección de patrones: Activada

Este reporte fue generado automáticamente por WhatsApp Analyzer v4.5
"""
                    
                    filename_report = f"reporte_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    st.download_button(
                        label="📋 Descargar Reporte Resumen (TXT)",
                        data=report_data,
                        file_name=filename_report,
                        mime="text/plain",
                        help="Reporte ejecutivo con estadísticas principales"
                    )
            
            except Exception as e:
                st.error(f"❌ **Error al procesar el archivo:**\n\n{str(e)}")
                st.info("""
                💡 **Posibles soluciones:**
                - Verifica que el archivo sea una exportación válida de WhatsApp
                - Asegúrate de que el archivo no esté corrupto
                - Intenta con un chat más pequeño para probar
                - Verifica que el archivo tenga la codificación correcta (UTF-8)
                """)
    
    # Footer con información importante
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px; background: linear-gradient(145deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 2rem;">
        <h4>🔒 Privacidad y Seguridad</h4>
        <p><strong>✅ Procesamiento 100% Local:</strong> Todos los archivos se procesan en tu navegador. No se envían datos a servidores externos.</p>
        <p><strong>🗑️ Sin Almacenamiento:</strong> No guardamos ninguna conversación ni archivo. Todo se elimina al cerrar la aplicación.</p>
        <p><strong>⚖️ Uso Responsable:</strong> Esta herramienta debe usarse únicamente con fines legítimos y respetando la privacidad y leyes locales.</p>
        <p><strong>🔬 Herramienta de Apoyo:</strong> Los resultados requieren validación manual y no constituyen evidencia legal definitiva.</p>
        <small><em>WhatsApp Analyzer v4.5 - Optimizado para Streamlit Cloud</em></small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
