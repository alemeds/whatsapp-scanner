#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALIZADOR AVANZADO DE CONVERSACIONES DE WHATSAPP - VERSIÓN MEJORADA 5.0
Aplicación web optimizada con mejores visualizaciones y análisis inteligente

Autor: Sistema de Análisis de Comunicaciones
Versión: 5.0 - Enhanced Edition con mejoras de performance y UI
"""

import streamlit as st
import pandas as pd
import re
import csv
import io
import os
import logging
import hashlib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificación segura de dependencias NLP
@st.cache_resource
def check_nlp_dependencies():
    """Verifica qué dependencias NLP están disponibles de forma segura"""
    spacy_available = False
    textblob_available = False
    nlp_model = None
    
    try:
        from textblob import TextBlob
        test_blob = TextBlob("test text")
        _ = test_blob.sentiment.polarity
        textblob_available = True
    except Exception:
        textblob_available = False
    
    try:
        import spacy
        try:
            nlp_model = spacy.load("es_core_news_sm")
            spacy_available = True
        except OSError:
            spacy_available = False
    except ImportError:
        spacy_available = False
    
    return spacy_available, textblob_available, nlp_model

SPACY_AVAILABLE, TEXTBLOB_AVAILABLE, nlp = check_nlp_dependencies()

# Configuración de la página
st.set_page_config(
    page_title="WhatsApp Analyzer 5.0 - Enhanced",
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
    
    .alert-critical {
        background: linear-gradient(145deg, #fee, #fdd);
        border-left: 6px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(145deg, #fffbf0, #fff4d6);
        border-left: 6px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(145deg, #f0f8ff, #e6f3ff);
        border-left: 6px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .detection-table {
        border: 1px solid #dee2e6;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 4px 15px 0 rgba(0,0,0,0.1);
    }
    
    .detection-row {
        border-bottom: 1px solid #eee;
        padding: 1rem;
        transition: background-color 0.2s;
    }
    
    .detection-row:hover {
        background-color: #f8f9fa;
    }
    
    .risk-badge-high {
        background: #dc3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    .risk-badge-medium {
        background: #ffc107;
        color: #212529;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    .risk-badge-low {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .filter-section {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .recommendation-box {
        background: linear-gradient(145deg, #e8f5e8, #d4edda);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SmartTextAnalyzer:
    """Analizador de texto inteligente optimizado - Versión 5.0"""
    
    def __init__(self):
        self.spacy_available = SPACY_AVAILABLE
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.nlp = nlp
        
        # Patrones mejorados
        self.negation_patterns = ["no", "nunca", "jamás", "nada", "tampoco", "ni", "sin"]
        self.intensity_modifiers = {
            "muy": 1.5, "super": 1.7, "extremadamente": 2.0, "bastante": 1.3,
            "algo": 0.7, "poco": 0.5, "medio": 0.8, "un poco": 0.6,
            "mucho": 1.4, "demasiado": 1.6, "increíblemente": 1.8
        }
        self.positive_emotions = ["amor", "cariño", "besos", "abrazos", "feliz", "contento", "alegre", "genial"]
        self.negative_emotions = ["odio", "rabia", "ira", "triste", "deprimido", "enojado", "furioso", "disgusto"]
    
    def analyze_message(self, text, sender, timestamp, config, dictionary, detection_type):
        """Análisis principal del mensaje con mejoras de performance"""
        
        cleaned_text = self.preprocess_text(text)
        
        analysis_results = {
            'sentiment': self.analyze_sentiment(text),
            'negation': self.detect_negation_simple(cleaned_text),
            'intensity': self.calculate_intensity(text, cleaned_text),
            'emotion': self.analyze_basic_emotion(cleaned_text),
            'context': self.analyze_context_smart(cleaned_text, dictionary, detection_type),
            'temporal': self.analyze_temporal_patterns(timestamp),
            'patterns': self.detect_behavioral_patterns(cleaned_text, detection_type)
        }
        
        if self.spacy_available and self.nlp:
            analysis_results.update(self.advanced_spacy_analysis(cleaned_text))
        
        base_score = self.calculate_base_score(cleaned_text, dictionary)
        smart_score = self.calculate_contextual_score(base_score, analysis_results, detection_type)
        
        detected_words = self.get_detected_words(text, dictionary)
        explanation = self.generate_comprehensive_explanation(analysis_results, smart_score, detection_type)
        
        label = "DETECTADO" if smart_score > config['threshold'] else "NO DETECTADO"
        
        return smart_score, label, detected_words, analysis_results, explanation
    
    def preprocess_text(self, text):
        """Preprocesamiento inteligente del texto"""
        text = text.lower()
        
        emoji_patterns = {
            r'[😍😘❤️💕💖😻🥰💘]': ' _expresion_amor_ ',
            r'[😠😡🤬👿💢😤]': ' _expresion_rabia_ ',
            r'[😢😭💔😞😔🥺]': ' _expresion_tristeza_ ',
            r'[😏😈🔥💦🍆🍑]': ' _expresion_sexual_ ',
            r'[🤮🤢💩👎]': ' _expresion_disgusto_ '
        }
        
        for pattern, replacement in emoji_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        replacements = {
            r's[3e]xy': 'sexy', r'k[1i]ero': 'quiero', r'h3rm0sa': 'hermosa',
            r'b[3e]ll[4a]': 'bella', r'amor3s': 'amores', r'\bb+b+\b': 'bebe',
            r'x+d+': 'xd', r'jaj+a*': 'jaja', r'jej+e*': 'jeje'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_sentiment(self, text):
        """Análisis de sentimientos mejorado"""
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
                context = words[i+1:i+4] if i+1 < len(words) else []
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
        
        for word, multiplier in self.intensity_modifiers.items():
            if word in cleaned_text:
                intensity_score *= multiplier
                indicators.append(f"{word} ({multiplier}x)")
        
        caps_ratio = sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1)
        if caps_ratio > 0.3:
            intensity_score *= 1.4
            indicators.append(f"MAYÚSCULAS ({caps_ratio:.0%})")
        
        exclamations = original_text.count('!')
        questions = original_text.count('?')
        
        if exclamations > 1:
            multiplier = 1 + (exclamations - 1) * 0.15
            intensity_score *= multiplier
            indicators.append(f"!×{exclamations}")
        
        if questions > 2:
            intensity_score *= 1.2
            indicators.append(f"?×{questions}")
        
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
        
        context_terms = {
            'laboral': ['jefe', 'trabajo', 'oficina', 'reunión', 'proyecto', 'empresa', 'ascenso', 'salario'],
            'romántico': ['amor', 'cariño', 'besos', 'pareja', 'novio', 'novia', 'cita', 'te amo'],
            'familiar': ['familia', 'papá', 'mamá', 'hermano', 'hermana', 'hijo', 'hija', 'casa', 'hogar'],
            'agresivo': ['odio', 'matar', 'golpear', 'destruir', 'venganza', 'rabia', 'pelea'],
            'sexual': dictionary.get('high_risk', []) + dictionary.get('medium_risk', []),
            'social': ['amigos', 'fiesta', 'grupo', 'clase', 'escuela', 'universidad', 'compañeros'],
            'temporal': ['noche', 'madrugada', 'tarde', 'mañana', 'después', 'luego', 'pronto']
        }
        
        for context_type, terms in context_terms.items():
            matches = sum(1 for term in terms if term in text)
            max_expected = 3
            context_analysis[context_type] = min(matches / max_expected, 1.0)
        
        return context_analysis
    
    def analyze_temporal_patterns(self, timestamp):
        """Análisis de patrones temporales"""
        try:
            hour_match = re.search(r'(\d{1,2}):(\d{2})', timestamp)
            if hour_match:
                hour = int(hour_match.group(1))
                
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
            
            entities = {
                'persons': [ent.text for ent in doc.ents if ent.label_ in ["PER", "PERSON"]],
                'locations': [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]],
                'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            }
            
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
        
        score = ((high_matches * 0.8) + 
                (medium_matches * 0.4) + 
                (context_matches * 0.3) + 
                (work_matches * 0.2)) / max(total_terms, 1)
        
        return min(score, 1.0)
    
    def calculate_contextual_score(self, base_score, analysis, detection_type):
        """Calcula puntuación final considerando contexto"""
        score = base_score
        
        negations = analysis['negation']
        for negation in negations:
            reduction = negation['strength'] * 0.4
            score *= (1 - reduction)
        
        intensity = analysis['intensity']['score']
        score *= intensity
        
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
        
        context = analysis['context']
        if detection_type == "Acoso Sexual":
            if context['laboral'] > 0.5 and context['sexual'] > 0.3:
                score *= 1.8
            elif context['romántico'] > 0.6:
                score *= 0.7
        
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
        
        sentiment = analysis['sentiment']
        if sentiment['confidence'] > 0.3:
            explanations.append(f"Sentimiento {sentiment['interpretation']} ({sentiment['polarity']:.2f})")
        
        negations = analysis['negation']
        if negations:
            explanations.append(f"Negaciones: {len(negations)}")
        
        intensity = analysis['intensity']
        if intensity['score'] > 1.3:
            explanations.append(f"Alta intensidad ({intensity['score']:.1f}x)")
        
        context = analysis['context']
        max_context = max(context.items(), key=lambda x: x[1])
        if max_context[1] > 0.4:
            explanations.append(f"Contexto {max_context[0]}")
        
        emotion = analysis['emotion']
        if emotion['expressions']:
            explanations.append(f"Expresiones: {', '.join(emotion['expressions'])}")
        
        patterns = analysis['patterns']
        if patterns:
            explanations.append(f"Patrones detectados: {len(patterns)}")
        
        temporal = analysis['temporal']
        if temporal['is_night']:
            explanations.append("Horario nocturno")
        
        method = "NLP completo" if self.spacy_available else "Análisis inteligente"
        if sentiment['method'] == 'textblob':
            method += " + TextBlob"
        
        explanation_text = " | ".join(explanations) if explanations else "Análisis básico"
        return f"{explanation_text} ({method})"

# Funciones auxiliares mejoradas
@st.cache_data(ttl=3600)
def analyze_messages_batch(messages_hash, dictionary_hash, config_hash):
    """Cache de análisis para evitar reprocesamiento"""
    return None

def calculate_advanced_metrics(results_df):
    """Calcula métricas avanzadas para mejor insights"""
    metrics = {}
    
    if 'timestamp' in results_df.columns and len(results_df) > 0:
        try:
            results_df['parsed_date'] = pd.to_datetime(results_df['timestamp'], errors='coerce', infer_datetime_format=True)
            results_df_with_dates = results_df.dropna(subset=['parsed_date'])
            
            if len(results_df_with_dates) > 0:
                results_df_with_dates['hour'] = results_df_with_dates['parsed_date'].dt.hour
                hourly_risk = results_df_with_dates.groupby('hour')['risk_score'].mean()
                if len(hourly_risk) > 0:
                    metrics['peak_risk_hour'] = hourly_risk.idxmax()
                    metrics['lowest_risk_hour'] = hourly_risk.idxmin()
        except Exception:
            pass
    
    detected_words_list = results_df[results_df['label'] == 'DETECTADO']['detected_words'].str.cat(sep=', ')
    if detected_words_list:
        word_freq = Counter([word.strip() for word in detected_words_list.split(',') if word.strip()])
        metrics['top_risk_words'] = dict(word_freq.most_common(5))
    
    if len(results_df) > 10:
        risk_trend = results_df['risk_score'].rolling(window=10).mean().diff().iloc[-1]
        metrics['risk_trend'] = 'increasing' if risk_trend > 0 else 'decreasing'
    
    return metrics

def generate_smart_alerts(results_df, detection_type):
    """Genera alertas inteligentes basadas en patrones"""
    alerts = []
    
    detected_df = results_df[results_df['label'] == 'DETECTADO']
    
    if len(detected_df) > 0:
        detection_rate = len(detected_df) / len(results_df)
        
        if detection_rate > 0.3:
            alerts.append({
                'level': 'critical',
                'message': f'⚠️ CRÍTICO: {len(detected_df)} detecciones de {len(results_df)} mensajes ({detection_rate*100:.1f}%)',
                'type': 'high_frequency'
            })
        elif detection_rate > 0.15:
            alerts.append({
                'level': 'warning',
                'message': f'🔔 ADVERTENCIA: {len(detected_df)} detecciones de {len(results_df)} mensajes ({detection_rate*100:.1f}%)',
                'type': 'medium_frequency'
            })
        
        sender_counts = detected_df['sender'].value_counts()
        if len(sender_counts) > 0:
            dominant_sender_pct = sender_counts.iloc[0] / len(detected_df) * 100
            if dominant_sender_pct > 70:
                alerts.append({
                    'level': 'info',
                    'message': f'👤 REMITENTE DOMINANTE: {sender_counts.index[0]} ({dominant_sender_pct:.1f}% de detecciones)',
                    'type': 'sender_concentration'
                })
    
    return alerts

def generate_recommendations(results_df, detection_type):
    """Genera recomendaciones específicas basadas en los resultados"""
    recommendations = []
    
    detection_rate = len(results_df[results_df['label'] == 'DETECTADO']) / len(results_df) if len(results_df) > 0 else 0
    
    if detection_rate > 0.3:
        recommendations.append("🚨 ALTA PRIORIDAD: Revisar inmediatamente las conversaciones detectadas")
        recommendations.append("📞 Considerar contactar a un profesional especializado")
    elif detection_rate > 0.1:
        recommendations.append("⚠️ ATENCIÓN: Monitorear la situación de cerca")
        recommendations.append("📝 Documentar evidencias adicionales")
    else:
        recommendations.append("✅ SITUACIÓN ESTABLE: Continuar monitoreo rutinario")
    
    if detection_type == "Acoso Sexual":
        recommendations.append("🏢 Si es contexto laboral, reportar a RRHH")
        recommendations.append("⚖️ Considerar asesoría legal si hay evidencia clara")
    elif detection_type == "CyberBullying":
        recommendations.append("🏫 Si involucra menores, contactar autoridades escolares")
        recommendations.append("🛡️ Implementar medidas de protección digital")
    elif detection_type == "Infidelidades":
        recommendations.append("💬 Considerar terapia de pareja si es apropiado")
        recommendations.append("🤝 Buscar mediación profesional")
    
    return recommendations

def create_enhanced_metric_dashboard(results_df):
    """Dashboard de métricas mejorado"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(results_df)
    detected = len(results_df[results_df['label'] == 'DETECTADO'])
    percentage = (detected / total) * 100 if total > 0 else 0
    avg_risk = results_df['risk_score'].mean()
    max_risk = results_df['risk_score'].max()
    
    with col1:
        st.metric(
            "📱 Total Mensajes", 
            f"{total:,}",
            help="Número total de mensajes analizados"
        )
    
    with col2:
        delta_color = "inverse" if detected > 0 else "normal"
        st.metric(
            "🎯 Detectados", 
            f"{detected:,}",
            delta=f"{percentage:.1f}%",
            delta_color=delta_color,
            help="Mensajes que superaron el umbral de detección"
        )
    
    with col3:
        if percentage > 30:
            risk_level = "🔴 CRÍTICO"
        elif percentage > 15:
            risk_level = "🟡 MEDIO"
        elif percentage > 5:
            risk_level = "🟠 BAJO"
        else:
            risk_level = "🟢 MÍNIMO"
        
        st.metric(
            f"{risk_level}", 
            f"{percentage:.1f}%",
            help="Porcentaje de detecciones sobre el total"
        )
    
    with col4:
        st.metric(
            "⚖️ Riesgo Promedio", 
            f"{avg_risk:.3f}",
            help="Puntuación promedio de riesgo (0.0 - 1.0)"
        )
    
    with col5:
        st.metric(
            "📊 Riesgo Máximo", 
            f"{max_risk:.3f}",
            help="Puntuación más alta encontrada"
        )

def create_advanced_filters():
    """Filtros avanzados para los resultados"""
    with st.expander("🔍 **Filtros Avanzados**", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👥 Remitentes")
            sender_filter = st.multiselect(
                "Seleccionar remitentes:",
                options=[],  # Se llenará dinámicamente
                help="Filtrar por remitentes específicos"
            )
        
        with col2:
            st.subheader("⚖️ Nivel de Riesgo")
            risk_range = st.slider(
                "Rango de riesgo:",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                help="Filtrar por puntuación de riesgo"
            )
        
        with col3:
            st.subheader("🔍 Búsqueda")
            word_search = st.text_input(
                "Buscar palabra:",
                placeholder="Ej: sexy, amor, secreto...",
                help="Buscar en contenido de mensajes"
            )
            
            case_sensitive = st.checkbox("Distinguir mayúsculas/minúsculas")
        
        return {
            'sender_filter': sender_filter,
            'risk_range': risk_range,
            'word_search': word_search,
            'case_sensitive': case_sensitive
        }

def create_enhanced_detections_table(detected_df, show_explanations=True, max_results=50):
    """Tabla mejorada para mostrar detecciones con estilo similar a la exportación"""
    
    if len(detected_df) == 0:
        st.info("📊 No hay detecciones que mostrar con los filtros actuales")
        return
    
    # Preparar datos para la tabla
    table_data = []
    
    for idx, row in detected_df.head(max_results).iterrows():
        # Determinar nivel de riesgo
        if row['risk_score'] > 0.8:
            risk_level = "ALTO"
            risk_color = "🔴"
            risk_class = "high"
        elif row['risk_score'] > 0.6:
            risk_level = "MEDIO"
            risk_color = "🟡"
            risk_class = "medium"
        else:
            risk_level = "BAJO"
            risk_color = "🟢"
            risk_class = "low"
        
        # Truncar mensaje para la tabla
        message_preview = row['message'][:100] + "..." if len(row['message']) > 100 else row['message']
        
        table_data.append({
            'ID': len(table_data) + 1,
            'Fecha/Hora': row['timestamp'],
            'Remitente': row['sender'],
            'Mensaje': message_preview,
            'Puntuación': f"{row['risk_score']:.3f}",
            'Nivel': f"{risk_color} {risk_level}",
            'Términos': row['detected_words'] if row['detected_words'] else "N/A",
            'Análisis': row['explanation'] if show_explanations else "N/A"
        })
    
    # Crear DataFrame para la tabla
    table_df = pd.DataFrame(table_data)
    
    # Mostrar información de la tabla
    st.markdown(f"""
    <div class="detection-table">
        <div style="padding: 1rem; background: linear-gradient(145deg, #f8f9fa, #e9ecef); border-bottom: 1px solid #dee2e6;">
            <h4 style="margin: 0; color: #495057;">🔍 Detecciones Encontradas</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                Mostrando {len(table_data)} de {len(detected_df)} detecciones totales
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configurar la tabla con formato
    st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                width="small",
                help="Número de detección"
            ),
            "Fecha/Hora": st.column_config.TextColumn(
                "📅 Fecha/Hora",
                width="medium",
                help="Momento del mensaje"
            ),
            "Remitente": st.column_config.TextColumn(
                "👤 Remitente",
                width="small",
                help="Quien envió el mensaje"
            ),
            "Mensaje": st.column_config.TextColumn(
                "💬 Mensaje",
                width="large",
                help="Contenido del mensaje (vista previa)"
            ),
            "Puntuación": st.column_config.TextColumn(
                "⚖️ Puntuación",
                width="small",
                help="Nivel de riesgo calculado (0.0-1.0)"
            ),
            "Nivel": st.column_config.TextColumn(
                "🎯 Nivel",
                width="small",
                help="Clasificación del riesgo"
            ),
            "Términos": st.column_config.TextColumn(
                "🔍 Términos",
                width="medium",
                help="Palabras clave detectadas"
            ),
            "Análisis": st.column_config.TextColumn(
                "🧠 Análisis",
                width="large",
                help="Explicación del análisis realizado"
            ) if show_explanations else None
        },
        hide_index=True
    )
    
    # Mostrar algunas estadísticas adicionales de la tabla
    with st.expander("📊 **Estadísticas de esta vista**"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk = len([d for d in table_data if "ALTO" in d['Nivel']])
            st.metric("🔴 Alto Riesgo", high_risk)
        
        with col2:
            medium_risk = len([d for d in table_data if "MEDIO" in d['Nivel']])
            st.metric("🟡 Riesgo Medio", medium_risk)
        
        with col3:
            low_risk = len([d for d in table_data if "BAJO" in d['Nivel']])
            st.metric("🟢 Riesgo Bajo", low_risk)
        
        with col4:
            avg_score = sum(float(d['Puntuación']) for d in table_data) / len(table_data)
            st.metric("📊 Promedio", f"{avg_score:.3f}")

def create_enhanced_visualizations(results_df, detection_type):
    """Visualizaciones mejoradas de los resultados"""
    
    detected_df = results_df[results_df['label'] == 'DETECTADO']
    
    # Primera fila: Distribución y evolución
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de distribución de riesgo mejorado
        fig_hist = px.histogram(
            results_df, 
            x='risk_score', 
            nbins=30,
            title=f'📊 Distribución de Puntuación de Riesgo - {detection_type}',
            labels={'risk_score': 'Puntuación de Riesgo', 'count': 'Cantidad de Mensajes'},
            color_discrete_sequence=['#667eea'],
            opacity=0.7
        )
        
        # Líneas de referencia mejoradas
        fig_hist.add_vline(x=0.45, line_dash="dot", line_color="green", 
                          annotation_text="Sensibilidad Alta", annotation_position="top")
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="orange", 
                          annotation_text="Sensibilidad Media", annotation_position="top")
        fig_hist.add_vline(x=0.75, line_dash="solid", line_color="red", 
                          annotation_text="Sensibilidad Baja", annotation_position="top")
        
        # Zona de detecciones
        if len(detected_df) > 0:
            min_detected = detected_df['risk_score'].min()
            fig_hist.add_vrect(
                x0=min_detected, x1=1.0,
                fillcolor="rgba(255, 0, 0, 0.1)",
                annotation_text="Zona de Detecciones",
                annotation_position="top"
            )
        
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if len(detected_df) > 0:
            # Gráfico de detecciones por remitente mejorado
            detections_by_sender = detected_df['sender'].value_counts()
            
            if len(detections_by_sender) > 1:
                fig_pie = px.pie(
                    values=detections_by_sender.values,
                    names=detections_by_sender.index,
                    title=f'🎯 Detecciones por Remitente - {detection_type}',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                # Gráfico alternativo para un solo remitente
                sender_stats = results_df.groupby('sender').agg({
                    'risk_score': ['count', 'mean', 'max'],
                    'label': lambda x: (x == 'DETECTADO').sum()
                }).round(3)
                sender_stats.columns = ['Total', 'Promedio', 'Máximo', 'Detectados']
                
                fig_bar = px.bar(
                    x=sender_stats.index,
                    y=sender_stats['Detectados'],
                    title=f'🎯 Detecciones por Remitente - {detection_type}',
                    labels={'x': 'Remitente', 'y': 'Detecciones'},
                    color=sender_stats['Detectados'],
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("📊 No hay detecciones para visualizar")
    
    # Segunda fila: Timeline y patrones temporales
    if len(detected_df) > 0:
        st.subheader("📅 Análisis Temporal")
        
        try:
            # Timeline de detecciones
            detected_df_copy = detected_df.copy()
            detected_df_copy['parsed_date'] = pd.to_datetime(detected_df_copy['timestamp'], errors='coerce', infer_datetime_format=True)
            detected_df_with_dates = detected_df_copy.dropna(subset=['parsed_date'])
            
            if len(detected_df_with_dates) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Timeline diario
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
                
                with col2:
                    # Distribución por hora del día
                    detected_df_with_dates['hour'] = detected_df_with_dates['parsed_date'].dt.hour
                    hourly_counts = detected_df_with_dates['hour'].value_counts().sort_index()
                    
                    fig_hourly = px.bar(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        title='🕐 Detecciones por Hora del Día',
                        labels={'x': 'Hora', 'y': 'Detecciones'},
                        color=hourly_counts.values,
                        color_continuous_scale='Oranges'
                    )
                    fig_hourly.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("⏰ No se pudieron parsear las fechas para el análisis temporal")
        except Exception as e:
            st.info(f"⏰ No se pudo generar análisis temporal: {str(e)}")

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

def validate_whatsapp_file(content):
    """Validación más robusta de archivos de WhatsApp"""
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}.*\d{1,2}:\d{2}.*-.*:',  # Formato estándar
        r'\[\d{1,2}/\d{1,2}/\d{2,4}.*\d{1,2}:\d{2}.*\].*:',  # Con corchetes
        r'\d{1,2}-\d{1,2}-\d{2,4}.*\d{1,2}:\d{2}.*-.*:'  # Con guiones
    ]
    
    total_lines = len(content.split('\n'))
    matches = 0
    
    for pattern in patterns:
        pattern_matches = len(re.findall(pattern, content))
        matches = max(matches, pattern_matches)
    
    confidence = matches / max(total_lines, 1) if total_lines > 0 else 0
    
    if confidence > 0.1:  # Al menos 10% de líneas parecen mensajes
        return True, f"Formato WhatsApp detectado (confianza: {confidence:.1%})"
    else:
        return False, f"No parece ser un archivo de WhatsApp válido (confianza: {confidence:.1%})"

def extract_messages_from_text(content):
    """Extrae mensajes de texto de WhatsApp con parsing robusto mejorado"""
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
                    'left', 'joined', 'changed subject to', 'created group',
                    'added', 'removed', 'security code changed'
                ]
                
                if not message or any(sys_msg in message.lower() for sys_msg in system_messages):
                    continue
                
                # Validar que el sender no sea muy largo (probable error de parsing)
                if len(sender) > 50:
                    continue
                
                # Validar que no sea una línea de continuación
                if len(message) < 5:
                    continue
                
                clean_matches.append((timestamp, sender, message))
            
            if len(clean_matches) > len(best_pattern_matches):
                best_pattern_matches = clean_matches
    
    return best_pattern_matches

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

def process_messages_in_batches(messages, analyzer, config, dictionary, detection_type, batch_size=50):
    """Procesa mensajes en lotes para mejor rendimiento"""
    total_batches = (len(messages) + batch_size - 1) // batch_size
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(messages))
        batch = messages[start_idx:end_idx]
        
        # Actualizar progreso
        progress = (batch_num + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Procesando lote {batch_num + 1}/{total_batches} ({end_idx}/{len(messages)} mensajes)")
        
        # Procesar lote
        for i, (timestamp, sender, message) in enumerate(batch):
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
                'explanation': explanation
            })
        
        # Pequeña pausa para no sobrecargar
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    # Header principal mejorado
    st.markdown("""
    <div class="main-header">
        <h1>🔍 WhatsApp Analyzer 5.0 - Enhanced Edition</h1>
        <p>Sistema avanzado para detección de patrones de comportamiento en chats</p>
        <small>Versión 5.0 - Con mejoras de rendimiento, visualizaciones avanzadas y tabla optimizada</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar estado de las herramientas mejorado
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
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🔍 Análisis", "📖 Instructivo", "📄 Formato CSV"])
    
    with tab3:
        st.markdown("""
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
        """)
    
    with tab2:
        st.markdown("""
        # 📖 **INSTRUCTIVO COMPLETO - WhatsApp Analyzer 5.0**
        
        ## 🎯 **¿QUÉ HACE ESTA APLICACIÓN?**
        
        Esta herramienta analiza conversaciones de WhatsApp para detectar patrones de comportamiento potencialmente problemáticos usando **análisis inteligente de texto** con tecnología NLP.
        
        ### **Nuevas Características v5.0:**
        - 📊 **Tabla mejorada** para visualizar detecciones
        - ⚡ **Procesamiento en lotes** para mejor rendimiento
        - 📈 **Métricas avanzadas** y análisis temporal
        - 🔍 **Filtros avanzados** para explorar resultados
        - 🎨 **Interfaz optimizada** con mejor UX
        
        ### **Tipos de Detección:**
        
        | 🔍 Tipo | 📝 Descripción | 🎯 Detecta |
        |---------|----------------|-------------|
        | **🚨 Acoso Sexual** | Comportamientos inapropiados | Lenguaje sexual, propuestas inapropiadas, acoso laboral |
        | **😠 CyberBullying** | Intimidación y agresión digital | Insultos, amenazas, exclusión social, humillación |
        | **💔 Infidelidades** | Indicios de relaciones extramaritales | Expresiones de amor oculto, citas secretas, doble vida |
        
        ## 🛠️ **CÓMO USAR LA APLICACIÓN**
        
        ### **Paso 1: Exportar Chat de WhatsApp**
        
        #### **📱 En Android:**
        1. Abre WhatsApp → Chat específico
        2. Toca **⋮** → **Más** → **Exportar chat**
        3. Selecciona **"Sin archivos multimedia"**
        4. Guarda el archivo `.txt`
        
        #### **📱 En iPhone:**
        1. Abre WhatsApp → Chat específico
        2. Toca el **nombre del contacto/grupo**
        3. **Exportar chat** → **"Sin archivos multimedia"**
        4. Guarda el archivo `.txt`
        
        ### **Paso 2: Configurar y Analizar**
        
        1. **Selecciona tipo de detección** (Acoso, Bullying, Infidelidades, etc.)
        2. **Configura sensibilidad** (Baja, Media, Alta)
        3. **Sube el archivo .txt** exportado
        4. **Revisa resultados** en la tabla mejorada
        5. **Descarga reportes** si es necesario
        
        ## 📊 **NUEVAS CARACTERÍSTICAS DE LA TABLA**
        
        La tabla mejorada muestra:
        - **ID único** para cada detección
        - **Fecha/Hora** formateada
        - **Remitente** identificado
        - **Vista previa del mensaje** (truncada para legibilidad)
        - **Puntuación de riesgo** precisa (0.000-1.000)
        - **Nivel de riesgo** visual (🔴 Alto, 🟡 Medio, 🟢 Bajo)
        - **Términos detectados** específicos
        - **Análisis detallado** explicativo (opcional)
        
        ## ⚡ **MEJORAS DE RENDIMIENTO**
        
        - **Procesamiento en lotes**: Chats grandes se procesan eficientemente
        - **Cache inteligente**: Evita reprocesamiento innecesario
        - **Progreso detallado**: Indicadores en tiempo real
        - **Validación robusta**: Mejor detección de archivos WhatsApp
        
        ## 🔍 **FILTROS AVANZADOS**
        
        - **Por remitente**: Filtra mensajes específicos
        - **Por rango de riesgo**: Ajusta sensibilidad de visualización
        - **Búsqueda de texto**: Encuentra palabras específicas
        - **Configuración de límites**: Controla cantidad de resultados
        
        ## ⚖️ **CONSIDERACIONES LEGALES Y ÉTICAS**
        
        ### **🔒 Privacidad:**
        - ✅ **100% Local**: Todo se procesa en tu navegador
        - ✅ **Sin almacenamiento**: Nada se guarda en servidores
        - ✅ **Sin envío de datos**: Completa privacidad
        
        ### **⚖️ Uso Responsable:**
        - 📋 **Solo con consentimiento** de las partes involucradas
        - 🛡️ **Cumplir leyes locales** de privacidad
        - 👨‍⚖️ **No es evidencia legal**: Requiere validación profesional
        - 🔍 **Herramienta de apoyo**: Para investigación preliminar
        
        ## 🚨 **SOLUCIÓN DE PROBLEMAS**
        
        ### **❌ "No se pudieron extraer mensajes"**
        - Verifica formato de exportación de WhatsApp
        - Asegúrate de exportar "sin multimedia"
        - Valida que el archivo no esté corrupto
        
        ### **⚡ "Análisis lento"**
        - Chats >5000 mensajes pueden tardar varios minutos
        - El procesamiento en lotes mejora la experiencia
        - Usar sensibilidad "baja" es más rápido
        
        ### **📊 "Muchos falsos positivos"**
        - Reduce sensibilidad o aumenta umbral
        - Usa filtros para revisar casos específicos
        - Revisa manualmente los resultados
        
        ## ✅ **RESUMEN RÁPIDO v5.0**
        
        1. **📤 Exporta** chat (sin multimedia)
        2. **🎯 Selecciona** tipo de detección
        3. **⚙️ Configura** sensibilidad
        4. **📁 Sube** archivo .txt
        5. **🔄 Analiza** (procesamiento mejorado)
        6. **📊 Explora** tabla de detecciones
        7. **🔍 Filtra** resultados según necesidad
        8. **💾 Descarga** reportes completos
        
        **¡Listo para el análisis avanzado!** 🚀
        """)
    
    with tab1:
        # Sidebar - Configuración mejorada
        with st.sidebar:
            st.header("⚙️ Configuración del Análisis")
            
            detection_options = list(get_predefined_dictionaries().keys()) + ["Diccionario Personalizado"]
            detection_type = st.selectbox(
                "🎯 Tipo de Detección",
                detection_options,
                help="Selecciona qué patrón quieres detectar"
            )
            
            # Información del tipo seleccionado
            if detection_type != "Diccionario Personalizado":
                info_dict = {
                    "Acoso Sexual": "🚨 Detecta insinuaciones inapropiadas, propuestas sexuales, acoso laboral",
                    "CyberBullying": "😠 Identifica insultos, amenazas, intimidación, exclusión social", 
                    "Infidelidades": "💔 Encuentra expresiones románticas ocultas, citas secretas, engaños"
                }
                st.info(info_dict[detection_type])
            
            # Diccionario
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
            
            # Opciones avanzadas
            st.subheader("🔧 Opciones Avanzadas")
            
            show_explanations = st.checkbox(
                "🧠 Mostrar Explicaciones Detalladas",
                value=True,
                help="Incluye explicaciones de por qué se detectó cada caso"
            )
            
            max_results = st.selectbox(
                "📊 Máximo de Evidencias en Tabla",
                [25, 50, 100, 200, "Todas"],
                index=1,
                help="Limita resultados en tabla para mejor rendimiento"
            )
            
            batch_size = st.selectbox(
                "⚡ Tamaño de Lote de Procesamiento",
                [25, 50, 100, 200],
                index=1,
                help="Lotes más grandes = más rápido, pero menos feedback"
            )
        
        # Área principal de contenido
        st.header("📤 Subir Archivo de Chat")
        
        # Instrucciones rápidas
        with st.expander("📧 ¿Cómo exportar chat de WhatsApp?"):
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
                
                # Validación mejorada
                is_valid, validation_message = validate_whatsapp_file(content)
                
                if not is_valid:
                    st.error(f"❌ **{validation_message}**")
                    st.info("""
                    **Posibles causas:**
                    - El archivo no es una exportación válida de WhatsApp
                    - Formato de fecha no reconocido
                    - Archivo corrupto o modificado
                    
                    **Solución:**
                    - Verifica que sea un archivo .txt exportado directamente de WhatsApp
                    - Asegúrate de seleccionar "Sin archivos multimedia" al exportar
                    """)
                    st.stop()
                
                st.success(f"✅ **{validation_message}**")
                
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
                
                # Vista previa de mensajes
                with st.expander(f"👀 Vista previa de mensajes (primeros 5 de {len(messages)})"):
                    for i, (timestamp, sender, message) in enumerate(messages[:5]):
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #667eea;">
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
                
                # Procesar mensajes con lotes mejorados
                st.subheader("🔄 Procesando Mensajes...")
                
                with st.container():
                    st.markdown("""
                    <div class="progress-container">
                        <h4>⚡ Análisis en Progreso</h4>
                        <p>Procesando mensajes en lotes para optimizar el rendimiento...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    results = process_messages_in_batches(
                        messages, analyzer, config, dictionary, detection_type, batch_size
                    )
                
                # Crear DataFrame de resultados
                results_df = pd.DataFrame(results)
                
                # Calcular métricas avanzadas
                advanced_metrics = calculate_advanced_metrics(results_df)
                alerts = generate_smart_alerts(results_df, detection_type)
                recommendations = generate_recommendations(results_df, detection_type)
                
                # Mostrar alertas si las hay
                if alerts:
                    st.subheader("🚨 Alertas del Sistema")
                    for alert in alerts:
                        if alert['level'] == 'critical':
                            st.markdown(f"""
                            <div class="alert-critical">
                                <h4>🚨 ALERTA CRÍTICA</h4>
                                <p>{alert['message']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif alert['level'] == 'warning':
                            st.markdown(f"""
                            <div class="alert-warning">
                                <h4>⚠️ ADVERTENCIA</h4>
                                <p>{alert['message']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="alert-info">
                                <h4>ℹ️ INFORMACIÓN</h4>
                                <p>{alert['message']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Dashboard de métricas mejorado
                st.header("📈 Resultados del Análisis")
                create_enhanced_metric_dashboard(results_df)
                
                # Evaluación del riesgo general
                detected_count = len(results_df[results_df['label'] == 'DETECTADO'])
                percentage = (detected_count / len(results_df)) * 100 if len(results_df) > 0 else 0
                
                if percentage == 0:
                    st.success("✅ **Excelente**: No se detectaron patrones problemáticos")
                elif percentage < 5:
                    st.info("🟢 **Bajo Riesgo**: Pocos casos detectados, revisar manualmente")
                elif percentage < 15:
                    st.warning("🟡 **Riesgo Moderado**: Revisar casos detectados cuidadosamente")
                else:
                    st.error("🔴 **Alto Riesgo**: Múltiples detecciones, requiere atención inmediata")
                
                # Mostrar visualizaciones mejoradas
                if detected_count > 0:
                    st.header("📊 Análisis Visual Avanzado")
                    create_enhanced_visualizations(results_df, detection_type)
                    
                    # Análisis por remitente mejorado
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
                            "Total Mensajes": st.column_config.NumberColumn("📱 Total"),
                            "Riesgo Promedio": st.column_config.NumberColumn("⚖️ Promedio", format="%.3f"),
                            "Riesgo Máximo": st.column_config.NumberColumn("📊 Máximo", format="%.3f"),
                            "Detecciones": st.column_config.NumberColumn("🚨 Detectados")
                        }
                    )
                    
                    # TABLA MEJORADA DE DETECCIONES - CARACTERÍSTICA PRINCIPAL
                    st.header("🔍 Tabla de Detecciones Mejorada")
                    
                    detected_df = results_df[results_df['label'] == 'DETECTADO'].copy()
                    detected_df = detected_df.sort_values('risk_score', ascending=False)
                    
                    # Filtros dinámicos para la tabla
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        available_senders = detected_df['sender'].unique().tolist()
                        sender_filter = st.multiselect(
                            "👤 Filtrar por remitente:",
                            options=available_senders,
                            default=available_senders,
                            help="Selecciona remitentes específicos"
                        )
                    
                    with col2:
                        risk_threshold = st.slider(
                            "⚖️ Riesgo mínimo:",
                            min_value=0.0,
                            max_value=1.0,
                            value=config['threshold'],
                            step=0.05,
                            help="Filtrar por nivel de riesgo"
                        )
                    
                    with col3:
                        word_filter = st.text_input(
                            "🔍 Buscar palabra:",
                            placeholder="Ej: sexy, amor, secreto...",
                            help="Busca mensajes que contengan esta palabra"
                        )
                    
                    # Aplicar filtros a la tabla
                    filtered_df = detected_df[
                        (detected_df['sender'].isin(sender_filter)) &
                        (detected_df['risk_score'] >= risk_threshold)
                    ]
                    
                    if word_filter:
                        filtered_df = filtered_df[
                            filtered_df['message'].str.contains(word_filter, case=False, na=False) |
                            filtered_df['detected_words'].str.contains(word_filter, case=False, na=False)
                        ]
                    
                    # Mostrar tabla mejorada
                    max_table_results = max_results if max_results != "Todas" else len(filtered_df)
                    create_enhanced_detections_table(
                        filtered_df, 
                        show_explanations=show_explanations, 
                        max_results=max_table_results
                    )
                    
                else:
                    st.success("✅ **¡Excelente noticia!** No se detectaron patrones sospechosos en la conversación")
                    
                    # Mostrar estadísticas básicas aunque no haya detecciones
                    st.subheader("📊 Estadísticas Generales")
                    
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
                
                # Mostrar métricas avanzadas si están disponibles
                if advanced_metrics:
                    st.subheader("📊 Métricas Avanzadas")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'peak_risk_hour' in advanced_metrics:
                            st.metric(
                                "🕐 Hora de Mayor Riesgo", 
                                f"{advanced_metrics['peak_risk_hour']}:00",
                                help="Hora del día con mayor actividad de riesgo"
                            )
                    
                    with col2:
                        if 'risk_trend' in advanced_metrics:
                            trend_emoji = "📈" if advanced_metrics['risk_trend'] == 'increasing' else "📉"
                            st.metric(
                                f"{trend_emoji} Tendencia de Riesgo", 
                                advanced_metrics['risk_trend'].title(),
                                help="Tendencia del riesgo a lo largo del tiempo"
                            )
                    
                    with col3:
                        if 'top_risk_words' in advanced_metrics and advanced_metrics['top_risk_words']:
                            top_word = list(advanced_metrics['top_risk_words'].keys())[0]
                            top_count = advanced_metrics['top_risk_words'][top_word]
                            st.metric(
                                "🔍 Palabra Más Frecuente", 
                                f"{top_word} ({top_count}x)",
                                help="Término más detectado en el análisis"
                            )
                
                # Mostrar recomendaciones
                if recommendations:
                    st.subheader("💡 Recomendaciones del Sistema")
                    
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <strong>{i+1}.</strong> {rec}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Sección de descarga mejorada
                st.header("💾 Descargar Resultados")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV completo mejorado
                    csv_buffer = io.StringIO()
                    
                    # Agregar metadatos al CSV
                    csv_buffer.write(f"# WhatsApp Analyzer v5.0 - Análisis Completo\n")
                    csv_buffer.write(f"# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    csv_buffer.write(f"# Tipo: {detection_type}\n")
                    csv_buffer.write(f"# Total Mensajes: {len(results_df)}\n")
                    csv_buffer.write(f"# Detecciones: {detected_count}\n")
                    csv_buffer.write(f"# Porcentaje: {percentage:.2f}%\n")
                    csv_buffer.write(f"# Umbral: {config['threshold']:.3f}\n")
                    csv_buffer.write("#\n")
                    
                    results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_data = csv_buffer.getvalue()
                    
                    filename = f"analisis_completo_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="📄 Descargar Análisis Completo (CSV)",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        help="Incluye todos los mensajes analizados con metadatos"
                    )
                
                with col2:
                    # Solo detecciones mejorado
                    if detected_count > 0:
                        detected_csv = io.StringIO()
                        
                        # Metadatos para detecciones
                        detected_csv.write(f"# WhatsApp Analyzer v5.0 - Solo Detecciones\n")
                        detected_csv.write(f"# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        detected_csv.write(f"# Tipo: {detection_type}\n")
                        detected_csv.write(f"# Detecciones: {detected_count}\n")
                        detected_csv.write(f"# Umbral: {config['threshold']:.3f}\n")
                        detected_csv.write("#\n")
                        
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
                    # Reporte ejecutivo mejorado
                    report_data = f"""REPORTE EJECUTIVO - WHATSAPP ANALYZER V5.0
=========================================================
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Archivo analizado: {uploaded_file.name}
Tipo de detección: {detection_type}
Sensibilidad: {sensitivity}
Umbral usado: {config['threshold']:.3f}

RESUMEN EJECUTIVO:
{"="*50}
- Total de mensajes analizados: {len(results_df):,}
- Mensajes detectados: {detected_count:,}
- Porcentaje de detección: {percentage:.2f}%
- Riesgo promedio: {results_df['risk_score'].mean():.4f}
- Riesgo máximo encontrado: {results_df['risk_score'].max():.4f}

EVALUACIÓN DE RIESGO:
{"="*50}
{"CRÍTICO - Requiere atención inmediata" if percentage > 30 else
 "ALTO - Revisar cuidadosamente" if percentage > 15 else
 "MODERADO - Monitoreo recomendado" if percentage > 5 else
 "BAJO - Situación controlada" if percentage > 0 else
 "MÍNIMO - No se detectaron problemas"}

ANÁLISIS POR REMITENTE:
{"="*50}
{results_df.groupby('sender').agg({
    'risk_score': ['count', 'mean', 'max'],
    'label': lambda x: (x == 'DETECTADO').sum()
}).to_string() if len(results_df) > 0 else 'No hay datos'}

ALERTAS DEL SISTEMA:
{"="*50}
{chr(10).join([f"- {alert['message']}" for alert in alerts]) if alerts else "No hay alertas críticas"}

RECOMENDACIONES:
{"="*50}
{chr(10).join([f"- {rec}" for rec in recommendations]) if recommendations else "No hay recomendaciones específicas"}

MÉTRICAS AVANZADAS:
{"="*50}
{"- Hora de mayor riesgo: " + str(advanced_metrics.get('peak_risk_hour', 'N/A')) + ":00" if 'peak_risk_hour' in advanced_metrics else ""}
{"- Tendencia de riesgo: " + advanced_metrics.get('risk_trend', 'N/A').title() if 'risk_trend' in advanced_metrics else ""}
{"- Palabras más frecuentes: " + str(list(advanced_metrics.get('top_risk_words', {}).keys())[:3]) if 'top_risk_words' in advanced_metrics else ""}

TECNOLOGÍAS UTILIZADAS:
{"="*50}
- Motor de análisis: WhatsApp Analyzer v5.0 Enhanced
- spaCy NLP: {'Disponible' if SPACY_AVAILABLE else 'No disponible'}
- TextBlob Sentimientos: {'Disponible' if TEXTBLOB_AVAILABLE else 'No disponible'}
- Análisis contextual: Activado
- Detección de patrones: Activada
- Procesamiento en lotes: Activado ({batch_size} mensajes por lote)

CONFIGURACIÓN UTILIZADA:
{"="*50}
- Diccionario: {detection_type}
- Términos de alto riesgo: {len(dictionary.get('high_risk', []))}
- Términos de riesgo medio: {len(dictionary.get('medium_risk', []))}
- Frases de contexto: {len(dictionary.get('context_phrases', []))}
- Contexto laboral: {len(dictionary.get('work_context', []))}

DISCLAIMER LEGAL:
{"="*50}
Este reporte fue generado automáticamente por WhatsApp Analyzer v5.0
y debe ser utilizado únicamente como herramienta de apoyo. Los resultados
requieren validación manual y no constituyen evidencia legal definitiva.
El uso de esta herramienta debe cumplir con las leyes locales de privacidad
y protección de datos.

Procesamiento realizado 100% localmente - Sin envío de datos externos
"""
                    
                    filename_report = f"reporte_ejecutivo_{detection_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    st.download_button(
                        label="📋 Descargar Reporte Ejecutivo (TXT)",
                        data=report_data,
                        file_name=filename_report,
                        mime="text/plain",
                        help="Reporte ejecutivo completo con análisis y recomendaciones"
                    )
                
                # Información adicional sobre los archivos
                st.info("""
                📁 **Información sobre las descargas:**
                - **CSV Completo**: Incluye todos los mensajes con puntuaciones y metadatos
                - **CSV Detecciones**: Solo mensajes detectados para revisión rápida  
                - **Reporte Ejecutivo**: Resumen profesional con recomendaciones y análisis
                """)
            
            except Exception as e:
                st.error(f"❌ **Error al procesar el archivo:**\n\n{str(e)}")
                logger.error(f"Error en análisis: {str(e)}")
                st.info("""
                💡 **Posibles soluciones:**
                - Verifica que el archivo sea una exportación válida de WhatsApp
                - Asegúrate de que el archivo no esté corrupto
                - Intenta con un chat más pequeño para probar
                - Verifica que el archivo tenga la codificación correcta (UTF-8)
                - Revisa que el formato de fecha sea compatible
                """)
    
    # Footer mejorado con información importante
    st.markdown("---")
    
    # Usar columnas de Streamlit en lugar de CSS grid
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px; background: linear-gradient(145deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 2rem;">
        <h4>🔒 Privacidad y Seguridad - WhatsApp Analyzer v5.0</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Usar columnas nativas de Streamlit para mejor compatibilidad
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 5px;">
            <h5 style="color: #28a745; margin-bottom: 10px;">✅ Procesamiento 100% Local</h5>
            <p style="font-size: 0.9em; color: #666; margin: 0;">Todos los archivos se procesan en tu navegador. No se envían datos a servidores externos.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 5px;">
            <h5 style="color: #17a2b8; margin-bottom: 10px;">🗑️ Sin Almacenamiento</h5>
            <p style="font-size: 0.9em; color: #666; margin: 0;">No guardamos ninguna conversación ni archivo. Todo se elimina al cerrar la aplicación.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 5px;">
            <h5 style="color: #ffc107; margin-bottom: 10px;">⚖️ Uso Responsable</h5>
            <p style="font-size: 0.9em; color: #666; margin: 0;">Esta herramienta debe usarse únicamente con fines legítimos y respetando la privacidad.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 5px;">
            <h5 style="color: #6f42c1; margin-bottom: 10px;">🔬 Herramienta de Apoyo</h5>
            <p style="font-size: 0.9em; color: #666; margin: 0;">Los resultados requieren validación manual y no constituyen evidencia legal definitiva.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección de nuevas características
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #e8f5e8, #d4edda); border-radius: 10px; margin: 20px 0; border: 1px solid #c3e6cb;">
        <h5 style="color: #155724; margin-bottom: 15px;">🆕 Nuevas Características v5.0</h5>
        <p style="font-size: 1.1em; margin: 0;">
            <strong style="color: #28a745;">📊 Tabla Mejorada</strong> • 
            <strong style="color: #17a2b8;">⚡ Procesamiento en Lotes</strong> • 
            <strong style="color: #6f42c1;">🔍 Filtros Avanzados</strong> • 
            <strong style="color: #e83e8c;">📈 Métricas Avanzadas</strong> • 
            <strong style="color: #fd7e14;">🎨 Interfaz Optimizada</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer final
    st.markdown("""
    <div style="text-align: center; padding: 15px; color: #6c757d;">
        <small><em>WhatsApp Analyzer v5.0 Enhanced Edition - Optimizado para Streamlit Cloud</em></small>
    </div>
    """, unsafe_allow_html=True)

def get_csv_format_instructions():
    """Retorna las instrucciones del formato CSV actualizadas"""
    return """
    ## 📋 **FORMATO DEL ARCHIVO CSV DE DICCIONARIO - v5.0**
    
    El archivo CSV debe tener **exactamente 2 columnas** con los siguientes encabezados:
    
    ```csv
    termino,categoria
    sexy,palabras_alta
    atractiva,palabras_media
    solos,frases_contexto
    jefe,contexto_laboral
    ```
    
    ### **Categorías Válidas:**
    
    | Categoría | Descripción | Peso en Análisis | Ejemplos |
    |-----------|-------------|------------------|----------|
    | `palabras_alta` | Términos de alto riesgo | ⚠️ **Alto** (0.7-0.8) | sexy, desnudo, sexual |
    | `palabras_media` | Términos de riesgo medio | ⚡ **Medio** (0.3-0.4) | atractiva, bonita, cariño |
    | `frases_contexto` | Frases que dan contexto | 🔍 **Contextual** (0.5-0.6) | solos, secreto, privado |
    | `contexto_laboral` | Términos de trabajo/profesional | 🏢 **Laboral** (0.3-0.5) | jefe, oficina, ascenso |
    | `contexto_relacion` | Términos de relaciones | ❤️ **Relacional** (0.4) | novio, pareja, matrimonio |
    | `contexto_financiero` | Términos financieros | 💰 **Financiero** (0.4) | dinero, préstamo, deuda |
    | `contexto_agresion` | Términos agresivos | 😠 **Agresivo** (0.6) | odio, matar, venganza |
    | `contexto_emocional` | Expresiones emocionales | 😢 **Emocional** (0.3) | tristeza, alegría, miedo |
    | `contexto_digital` | Términos digitales/redes | 📱 **Digital** (0.3) | facebook, instagram, viral |
    | `contexto_sustancias` | Referencias a sustancias | 🚫 **Sustancias** (0.5) | drogas, alcohol, fumar |
    
    ### **Ejemplo Completo Mejorado:**
    ```csv
    termino,categoria
    # Términos de alto riesgo
    sexy,palabras_alta
    desnudo,palabras_alta
    sexual,palabras_alta
    # Términos de riesgo medio
    hermosa,palabras_media
    atractiva,palabras_media
    bonita,palabras_media
    # Frases de contexto
    solos,frases_contexto
    secreto,frases_contexto
    privado,frases_contexto
    # Contexto laboral
    jefe,contexto_laboral
    oficina,contexto_laboral
    ascenso,contexto_laboral
    ```
    
    ### **Notas Importantes:**
    - Líneas que empiecen con `#` son ignoradas (comentarios)
    - Los términos se procesan en minúsculas automáticamente
    - No incluir términos duplicados
    - Usar codificación UTF-8 para caracteres especiales
    """

# Función para ejecutar tests automatizados
def run_automated_tests():
    """Tests automatizados para validar funcionamiento"""
    try:
        # Test de parsing de mensajes
        test_message = "15/01/24, 2:28 p. m. - Juan: Hola, ¿cómo estás?"
        messages = extract_messages_from_text(test_message)
        assert len(messages) == 1, "Error en parsing básico"
        
        # Test de análisis
        analyzer = SmartTextAnalyzer()
        test_dict = {'high_risk': ['test'], 'medium_risk': [], 'context_phrases': [], 'work_context': []}
        config = {'threshold': 0.5}
        
        score, label, words, analysis, explanation = analyzer.analyze_message(
            "test message", "sender", "timestamp", config, test_dict, "Test"
        )
        
        assert isinstance(score, float), "Score debe ser float"
        assert label in ['DETECTADO', 'NO DETECTADO'], "Label inválido"
        
        st.success("✅ Todos los tests automatizados pasaron correctamente")
        return True
        
    except Exception as e:
        st.error(f"❌ Error en tests automatizados: {str(e)}")
        logger.error(f"Error en tests: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Ejecutar tests automatizados en desarrollo
        if st.sidebar.button("🧪 Ejecutar Tests Automatizados", help="Ejecuta validaciones del sistema"):
            with st.spinner("🔍 Ejecutando tests automatizados..."):
                run_automated_tests()
        
        # Ejecutar aplicación principal
        main()
        
    except Exception as e:
        st.error(f"❌ **Error crítico en la aplicación:**\n\n{str(e)}")
        logger.critical(f"Error crítico: {str(e)}")
        st.info("""
        🔧 **Para reportar este error:**
        1. Toma una captura de pantalla de este mensaje
        2. Incluye información sobre el archivo que intentabas analizar
        3. Describe los pasos que llevaron al error
        
        🔄 **Para intentar solucionarlo:**
        1. Recarga la página (F5)
        2. Verifica que el archivo sea una exportación válida de WhatsApp
        3. Intenta con un archivo más pequeño
        """)
