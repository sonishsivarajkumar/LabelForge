"""
NLP-focused labeling function templates.

This module provides common NLP labeling function templates including
sentiment analysis, topic classification, named entity recognition,
and other text processing utilities.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Set, Union, Callable
import warnings

from ..types import Example
from ..lf import lf

# Try to import optional NLP libraries
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


class SentimentLF:
    """
    Pre-built labeling functions for sentiment analysis.
    
    Provides simple rule-based sentiment classification using
    keyword lists and optional sentiment analysis libraries.
    """
    
    def __init__(self):
        """Initialize sentiment LF templates."""
        self.positive_keywords = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'love', 'perfect', 'outstanding', 'brilliant', 'superb',
            'happy', 'pleased', 'satisfied', 'delighted', 'thrilled', 'impressed'
        }
        
        self.negative_keywords = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'disappointing',
            'hate', 'worst', 'useless', 'pathetic', 'annoying', 'frustrating',
            'angry', 'upset', 'dissatisfied', 'unhappy', 'disappointed', 'regret'
        }
        
        self.intensifiers = {
            'very', 'extremely', 'really', 'absolutely', 'completely', 'totally',
            'incredibly', 'amazingly', 'truly', 'quite', 'so', 'too'
        }
        
        self.negators = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'nowhere',
            'neither', 'nor', 'barely', 'hardly', 'scarcely', 'seldom'
        }
    
    def create_keyword_sentiment_lf(
        self, 
        name: str = "keyword_sentiment",
        positive_label: int = 1,
        negative_label: int = 0,
        threshold: int = 1
    ) -> Callable[[Example], int]:
        """
        Create a keyword-based sentiment labeling function.
        
        Args:
            name: Name for the labeling function
            positive_label: Label for positive sentiment
            negative_label: Label for negative sentiment
            threshold: Minimum keyword count for classification
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def keyword_sentiment_lf(example: Example) -> int:
            text = example.text.lower()
            words = set(text.split())
            
            pos_count = len(words.intersection(self.positive_keywords))
            neg_count = len(words.intersection(self.negative_keywords))
            
            if pos_count >= threshold and pos_count > neg_count:
                return positive_label
            elif neg_count >= threshold and neg_count > pos_count:
                return negative_label
            else:
                return -1  # Abstain
        
        return keyword_sentiment_lf
    
    def create_textblob_sentiment_lf(
        self,
        name: str = "textblob_sentiment", 
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        positive_label: int = 1,
        negative_label: int = 0
    ) -> Callable[[Example], int]:
        """
        Create a TextBlob-based sentiment labeling function.
        
        Args:
            name: Name for the labeling function
            positive_threshold: Minimum polarity for positive sentiment
            negative_threshold: Maximum polarity for negative sentiment
            positive_label: Label for positive sentiment
            negative_label: Label for negative sentiment
            
        Returns:
            Labeling function
        """
        if not HAS_TEXTBLOB:
            warnings.warn("TextBlob not available. Install with: pip install textblob")
            return self.create_keyword_sentiment_lf(name, positive_label, negative_label)
        
        @lf(name=name)
        def textblob_sentiment_lf(example: Example) -> int:
            try:
                blob = TextBlob(example.text)
                polarity = blob.sentiment.polarity
                
                if polarity >= positive_threshold:
                    return positive_label
                elif polarity <= negative_threshold:
                    return negative_label
                else:
                    return -1  # Abstain
            except Exception:
                return -1  # Abstain on error
        
        return textblob_sentiment_lf
    
    def create_capitalization_sentiment_lf(
        self,
        name: str = "caps_sentiment",
        positive_label: int = 1,
        negative_label: int = 0,
        caps_threshold: float = 0.3
    ) -> Callable[[Example], int]:
        """
        Create sentiment LF based on capitalization patterns.
        
        Args:
            name: Name for the labeling function
            positive_label: Label for positive sentiment
            negative_label: Label for negative sentiment
            caps_threshold: Minimum ratio of capitalized words
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def caps_sentiment_lf(example: Example) -> int:
            words = example.text.split()
            if len(words) < 3:  # Too short to be reliable
                return -1
            
            caps_ratio = sum(1 for word in words if word.isupper()) / len(words)
            
            if caps_ratio >= caps_threshold:
                # High caps could indicate strong emotion
                text_lower = example.text.lower()
                pos_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
                neg_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
                
                if pos_count > neg_count:
                    return positive_label
                elif neg_count > pos_count:
                    return negative_label
            
            return -1  # Abstain
        
        return caps_sentiment_lf


class TopicLF:
    """
    Pre-built labeling functions for topic classification.
    
    Provides keyword-based and pattern-based topic classification
    for common domains and categories.
    """
    
    def __init__(self):
        """Initialize topic LF templates."""
        self.topic_keywords = {
            'technology': {
                'software', 'computer', 'digital', 'algorithm', 'programming',
                'tech', 'AI', 'machine learning', 'data', 'internet', 'app',
                'mobile', 'cloud', 'cybersecurity', 'blockchain', 'startup'
            },
            'business': {
                'company', 'market', 'sales', 'revenue', 'profit', 'investment',
                'finance', 'economy', 'corporate', 'business', 'CEO', 'strategy',
                'competition', 'customer', 'product', 'service', 'industry'
            },
            'health': {
                'health', 'medical', 'doctor', 'hospital', 'medicine', 'patient',
                'disease', 'treatment', 'therapy', 'wellness', 'fitness', 'diet',
                'nutrition', 'vaccine', 'pharmaceutical', 'clinical', 'diagnosis'
            },
            'sports': {
                'sport', 'game', 'team', 'player', 'match', 'win', 'score',
                'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
                'olympic', 'championship', 'coach', 'athlete', 'stadium', 'league'
            },
            'politics': {
                'government', 'election', 'political', 'president', 'congress',
                'senator', 'policy', 'law', 'vote', 'campaign', 'party',
                'democracy', 'republican', 'democrat', 'senate', 'house'
            }
        }
    
    def create_keyword_topic_lf(
        self,
        topic: str,
        label: int,
        name: Optional[str] = None,
        threshold: int = 2,
        custom_keywords: Optional[Set[str]] = None
    ) -> Callable[[Example], int]:
        """
        Create a keyword-based topic labeling function.
        
        Args:
            topic: Topic name (must be in topic_keywords or provide custom_keywords)
            label: Label to assign for this topic
            name: Custom name for the LF
            threshold: Minimum keyword matches required
            custom_keywords: Custom keyword set (overrides built-in)
            
        Returns:
            Labeling function
        """
        if custom_keywords:
            keywords = custom_keywords
        elif topic in self.topic_keywords:
            keywords = self.topic_keywords[topic]
        else:
            raise ValueError(f"Unknown topic '{topic}'. Provide custom_keywords or use: {list(self.topic_keywords.keys())}")
        
        lf_name = name or f"{topic}_keywords"
        
        @lf(name=lf_name)
        def keyword_topic_lf(example: Example) -> int:
            text_lower = example.text.lower()
            
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches >= threshold:
                return label
            else:
                return -1  # Abstain
        
        return keyword_topic_lf
    
    def create_multi_topic_lf(
        self,
        topic_labels: Dict[str, int],
        name: str = "multi_topic",
        threshold: int = 2
    ) -> Callable[[Example], int]:
        """
        Create a multi-topic labeling function.
        
        Args:
            topic_labels: Dictionary mapping topic names to labels
            name: Name for the labeling function
            threshold: Minimum keyword matches required
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def multi_topic_lf(example: Example) -> int:
            text_lower = example.text.lower()
            
            topic_scores = {}
            for topic, label in topic_labels.items():
                if topic in self.topic_keywords:
                    keywords = self.topic_keywords[topic]
                    score = sum(1 for keyword in keywords if keyword in text_lower)
                    if score >= threshold:
                        topic_scores[topic] = score
            
            if topic_scores:
                # Return label of topic with highest score
                best_topic = max(topic_scores, key=topic_scores.get)
                return topic_labels[best_topic]
            else:
                return -1  # Abstain
        
        return multi_topic_lf


class NERBasedLF:
    """
    Named Entity Recognition based labeling functions.
    
    Uses entity types and patterns to classify text based on
    the presence and types of named entities.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize NER-based LF templates.
        
        Args:
            model: SpaCy model name to use
        """
        self.model_name = model
        self.nlp = None
        
        if HAS_SPACY:
            try:
                import spacy
                self.nlp = spacy.load(model)
            except OSError:
                warnings.warn(f"SpaCy model '{model}' not found. Install with: python -m spacy download {model}")
                HAS_SPACY = False
    
    def create_entity_count_lf(
        self,
        entity_types: List[str],
        label: int,
        name: str,
        min_count: int = 1,
        max_count: Optional[int] = None
    ) -> Callable[[Example], int]:
        """
        Create LF based on entity count thresholds.
        
        Args:
            entity_types: List of entity types to count (e.g., ['PERSON', 'ORG'])
            label: Label to assign when conditions are met
            name: Name for the labeling function
            min_count: Minimum entity count
            max_count: Maximum entity count (optional)
            
        Returns:
            Labeling function
        """
        if not HAS_SPACY or self.nlp is None:
            warnings.warn("SpaCy not available. Returning abstaining LF.")
            
            @lf(name=name)
            def dummy_lf(example: Example) -> int:
                return -1
            
            return dummy_lf
        
        @lf(name=name)
        def entity_count_lf(example: Example) -> int:
            try:
                doc = self.nlp(example.text)
                entity_count = sum(1 for ent in doc.ents if ent.label_ in entity_types)
                
                if min_count <= entity_count:
                    if max_count is None or entity_count <= max_count:
                        return label
                
                return -1  # Abstain
            except Exception:
                return -1  # Abstain on error
        
        return entity_count_lf
    
    def create_entity_ratio_lf(
        self,
        entity_types: List[str],
        label: int,
        name: str,
        min_ratio: float = 0.1
    ) -> Callable[[Example], int]:
        """
        Create LF based on ratio of entities to total tokens.
        
        Args:
            entity_types: List of entity types to consider
            label: Label to assign when conditions are met
            name: Name for the labeling function
            min_ratio: Minimum ratio of entities to tokens
            
        Returns:
            Labeling function
        """
        if not HAS_SPACY or self.nlp is None:
            warnings.warn("SpaCy not available. Returning abstaining LF.")
            
            @lf(name=name)
            def dummy_lf(example: Example) -> int:
                return -1
            
            return dummy_lf
        
        @lf(name=name)
        def entity_ratio_lf(example: Example) -> int:
            try:
                doc = self.nlp(example.text)
                
                if len(doc) == 0:
                    return -1
                
                entity_count = sum(1 for ent in doc.ents if ent.label_ in entity_types)
                ratio = entity_count / len(doc)
                
                if ratio >= min_ratio:
                    return label
                else:
                    return -1  # Abstain
            except Exception:
                return -1  # Abstain on error
        
        return entity_ratio_lf


class KeywordLF:
    """
    Simple keyword-based labeling functions.
    
    Provides flexible keyword matching with various options
    for case sensitivity, word boundaries, and boolean logic.
    """
    
    @staticmethod
    def create_simple_keyword_lf(
        keywords: Union[str, List[str]],
        label: int,
        name: str,
        case_sensitive: bool = False,
        word_boundary: bool = True,
        require_all: bool = False
    ) -> Callable[[Example], int]:
        """
        Create a simple keyword-based labeling function.
        
        Args:
            keywords: Single keyword or list of keywords
            label: Label to assign when keywords are found
            name: Name for the labeling function
            case_sensitive: Whether to match case sensitively
            word_boundary: Whether to require word boundaries
            require_all: Whether all keywords must be present (AND vs OR)
            
        Returns:
            Labeling function
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        @lf(name=name)
        def simple_keyword_lf(example: Example) -> int:
            text = example.text if case_sensitive else example.text.lower()
            search_keywords = keywords if case_sensitive else [k.lower() for k in keywords]
            
            if word_boundary:
                # Use word boundary matching
                words = set(text.split())
                found_keywords = [k for k in search_keywords if k in words]
            else:
                # Use substring matching
                found_keywords = [k for k in search_keywords if k in text]
            
            if require_all:
                # All keywords must be present
                if len(found_keywords) == len(search_keywords):
                    return label
            else:
                # Any keyword can be present
                if found_keywords:
                    return label
            
            return -1  # Abstain
        
        return simple_keyword_lf


class RegexLF:
    """
    Regular expression based labeling functions.
    
    Provides utilities for pattern-based text classification
    using regular expressions.
    """
    
    @staticmethod
    def create_regex_lf(
        pattern: str,
        label: int,
        name: str,
        flags: int = re.IGNORECASE,
        min_matches: int = 1
    ) -> Callable[[Example], int]:
        """
        Create a regex-based labeling function.
        
        Args:
            pattern: Regular expression pattern
            label: Label to assign when pattern matches
            name: Name for the labeling function
            flags: Regex flags (default: case insensitive)
            min_matches: Minimum number of matches required
            
        Returns:
            Labeling function
        """
        compiled_pattern = re.compile(pattern, flags)
        
        @lf(name=name)
        def regex_lf(example: Example) -> int:
            matches = compiled_pattern.findall(example.text)
            
            if len(matches) >= min_matches:
                return label
            else:
                return -1  # Abstain
        
        return regex_lf
    
    @staticmethod
    def create_email_pattern_lf(
        label: int,
        name: str = "email_pattern"
    ) -> Callable[[Example], int]:
        """
        Create LF that detects email patterns.
        
        Args:
            label: Label for texts containing email patterns
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        return RegexLF.create_regex_lf(email_pattern, label, name)
    
    @staticmethod
    def create_phone_pattern_lf(
        label: int,
        name: str = "phone_pattern"
    ) -> Callable[[Example], int]:
        """
        Create LF that detects phone number patterns.
        
        Args:
            label: Label for texts containing phone patterns
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        phone_pattern = r'\\b(?:\\+?1[-.]?)?\\(?([0-9]{3})\\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\\b'
        return RegexLF.create_regex_lf(phone_pattern, label, name)
    
    @staticmethod
    def create_url_pattern_lf(
        label: int,
        name: str = "url_pattern"
    ) -> Callable[[Example], int]:
        """
        Create LF that detects URL patterns.
        
        Args:
            label: Label for texts containing URLs
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        url_pattern = r'https?://(?:[-\\w.])+(?:[:\\d]+)?(?:/(?:[\\w/_.])*(?:\\?(?:[\\w&=%.])*)?(?:#(?:[\\w.])*)?)?'
        return RegexLF.create_regex_lf(url_pattern, label, name)


class LengthLF:
    """
    Text length based labeling functions.
    
    Provides utilities for classifying text based on
    character count, word count, and sentence count.
    """
    
    @staticmethod
    def create_char_length_lf(
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        label: int = 1,
        name: str = "char_length"
    ) -> Callable[[Example], int]:
        """
        Create LF based on character length.
        
        Args:
            min_length: Minimum character count
            max_length: Maximum character count
            label: Label to assign when conditions are met
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def char_length_lf(example: Example) -> int:
            length = len(example.text)
            
            if min_length is not None and length < min_length:
                return -1
            if max_length is not None and length > max_length:
                return -1
            
            return label
        
        return char_length_lf
    
    @staticmethod
    def create_word_count_lf(
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        label: int = 1,
        name: str = "word_count"
    ) -> Callable[[Example], int]:
        """
        Create LF based on word count.
        
        Args:
            min_words: Minimum word count
            max_words: Maximum word count
            label: Label to assign when conditions are met
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def word_count_lf(example: Example) -> int:
            word_count = len(example.text.split())
            
            if min_words is not None and word_count < min_words:
                return -1
            if max_words is not None and word_count > max_words:
                return -1
            
            return label
        
        return word_count_lf
    
    @staticmethod
    def create_sentence_count_lf(
        min_sentences: Optional[int] = None,
        max_sentences: Optional[int] = None,
        label: int = 1,
        name: str = "sentence_count"
    ) -> Callable[[Example], int]:
        """
        Create LF based on sentence count.
        
        Args:
            min_sentences: Minimum sentence count
            max_sentences: Maximum sentence count
            label: Label to assign when conditions are met
            name: Name for the labeling function
            
        Returns:
            Labeling function
        """
        @lf(name=name)
        def sentence_count_lf(example: Example) -> int:
            # Simple sentence splitting on periods, exclamation marks, question marks
            sentences = re.split(r'[.!?]+', example.text.strip())
            sentence_count = len([s for s in sentences if s.strip()])
            
            if min_sentences is not None and sentence_count < min_sentences:
                return -1
            if max_sentences is not None and sentence_count > max_sentences:
                return -1
            
            return label
        
        return sentence_count_lf
