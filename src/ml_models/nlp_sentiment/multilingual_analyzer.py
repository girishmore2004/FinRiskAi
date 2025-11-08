"""
Multi-language sentiment analysis for Indian SMEs
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from textblob import TextBlob
import re
import logging
from langdetect import detect
from googletrans import Translator

logger = logging.getLogger(__name__)

class MultilingualSentimentAnalyzer:
    """Advanced sentiment analysis for multiple Indian languages"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.translator = Translator()
        
        # Initialize models for different languages
        self._initialize_models()
        
        # Language code mapping
        self.language_codes = {
            'hindi': 'hi',
            'english': 'en',
            'marathi': 'mr',
            'gujarati': 'gu',
            'tamil': 'ta',
            'telugu': 'te',
            'kannada': 'kn',
            'bengali': 'bn'
        }
    
    def _initialize_models(self):
        """Initialize pre-trained models for different languages"""
        
        # English sentiment analysis
        self.pipelines['english'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Hindi sentiment analysis
        try:
            self.pipelines['hindi'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
        except Exception as e:
            logger.warning(f"Could not load Hindi model: {e}")
            self.pipelines['hindi'] = None
        
        # Multilingual model as fallback
        self.pipelines['multilingual'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            detected_lang = detect(text)
            
            # Map detected language codes to our supported languages
            lang_mapping = {
                'en': 'english',
                'hi': 'hindi',
                'mr': 'marathi',
                'gu': 'gujarati',
                'ta': 'tamil',
                'te': 'telugu',
                'kn': 'kannada',
                'bn': 'bengali'
            }
            
            return lang_mapping.get(detected_lang, 'english')
        
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'english'  # Default to English
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep Devanagari and other Indian scripts
        text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        if not clean_text:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'language': 'unknown'}
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(clean_text)
        
        # Choose appropriate model
        if language in self.pipelines and self.pipelines[language] is not None:
            pipeline_model = self.pipelines[language]
        else:
            pipeline_model = self.pipelines['multilingual']
        
        try:
            # Get sentiment prediction
            result = pipeline_model(clean_text)[0]
            
            # Normalize labels
            sentiment_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral'
            }
            
            sentiment = sentiment_mapping.get(result['label'].upper(), result['label'].lower())
            confidence = result['score']
            
            return {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'language': language,
                'original_text': text,
                'processed_text': clean_text
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'language': language,
                'error': str(e)
            }
    
    def analyze_business_reputation(self, business_texts: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze business reputation from multiple text sources"""
        
        reputation_analysis = {
            'overall_sentiment': 'neutral',
            'confidence': 0.0,
            'source_analysis': {},
            'language_breakdown': {},
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'key_insights': []
        }
        
        all_sentiments = []
        all_confidences = []
        language_counts = {}
        
        # Analyze each source
        for source, texts in business_texts.items():
            source_sentiments = []
            source_confidences = []
            
            for text in texts:
                if not text or len(text.strip()) < 5:
                    continue
                
                sentiment_result = self.analyze_sentiment(text)
                source_sentiments.append(sentiment_result['sentiment'])
                source_confidences.append(sentiment_result['confidence'])
                all_sentiments.append(sentiment_result['sentiment'])
                all_confidences.append(sentiment_result['confidence'])
                
                # Count languages
                language = sentiment_result['language']
                language_counts[language] = language_counts.get(language, 0) + 1
            
            if source_sentiments:
                # Calculate source-level metrics
                positive_ratio = source_sentiments.count('positive') / len(source_sentiments)
                negative_ratio = source_sentiments.count('negative') / len(source_sentiments)
                neutral_ratio = source_sentiments.count('neutral') / len(source_sentiments)
                avg_confidence = np.mean(source_confidences)
                
                reputation_analysis['source_analysis'][source] = {
                    'sentiment_distribution': {
                        'positive': positive_ratio,
                        'negative': negative_ratio,
                        'neutral': neutral_ratio
                    },
                    'average_confidence': float(avg_confidence),
                    'total_texts': len(source_sentiments),
                    'dominant_sentiment': max(
                        ['positive', 'negative', 'neutral'],
                        key=lambda x: source_sentiments.count(x)
                    )
                }
        
        if all_sentiments:
            # Calculate overall metrics
            positive_count = all_sentiments.count('positive')
            negative_count = all_sentiments.count('negative')
            neutral_count = all_sentiments.count('neutral')
            total_count = len(all_sentiments)
            
            reputation_analysis['sentiment_distribution'] = {
                'positive': positive_count / total_count,
                'negative': negative_count / total_count,
                'neutral': neutral_count / total_count
            }
            
            reputation_analysis['overall_sentiment'] = max(
                ['positive', 'negative', 'neutral'],
                key=lambda x: all_sentiments.count(x)
            )
            
            reputation_analysis['confidence'] = float(np.mean(all_confidences))
            reputation_analysis['language_breakdown'] = language_counts
            
            # Generate insights
            reputation_analysis['key_insights'] = self._generate_reputation_insights(
                reputation_analysis
            )
        
        return reputation_analysis
    
    def _generate_reputation_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from reputation analysis"""
        insights = []
        
        sentiment_dist = analysis['sentiment_distribution']
        overall_sentiment = analysis['overall_sentiment']
        confidence = analysis['confidence']
        
        # Overall sentiment insight
        if confidence > 0.7:
            confidence_level = "high confidence"
        elif confidence > 0.5:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        insights.append(
            f"Overall business sentiment is {overall_sentiment} with {confidence_level} "
            f"(confidence: {confidence:.1%})"
        )
        
        # Sentiment distribution insights
        if sentiment_dist['positive'] > 0.6:
            insights.append("Strong positive reputation with majority of positive mentions")
        elif sentiment_dist['negative'] > 0.4:
            insights.append("Significant negative sentiment detected - reputation concerns")
        elif sentiment_dist['neutral'] > 0.7:
            insights.append("Mostly neutral mentions - limited strong opinions")
        
        # Source-specific insights
        source_analysis = analysis.get('source_analysis', {})
        for source, source_data in source_analysis.items():
            dominant = source_data['dominant_sentiment']
            if source_data['average_confidence'] > 0.7:
                insights.append(f"{source.title()} shows {dominant} sentiment with high confidence")
        
        # Language insights
        language_breakdown = analysis.get('language_breakdown', {})
        if language_breakdown:
            dominant_language = max(language_breakdown, key=language_breakdown.get)
            insights.append(f"Most mentions are in {dominant_language}")
        
        return insights
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> List[Dict[str, Any]]:
        """Extract key topics from business-related texts"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        if not texts or len(texts) < 2:
            return []
        
        # Preprocess texts
        clean_texts = [self.preprocess_text(text) for text in texts if text and len(text.strip()) > 10]
        
        if len(clean_texts) < 2:
            return []
        
        try:
            # Vectorize texts
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            doc_term_matrix = vectorizer.fit_transform(clean_texts)
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=min(num_topics, len(clean_texts)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': top_words[:5],
                    'weight': float(np.sum(topic))
                })
            
            return topics
        
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []