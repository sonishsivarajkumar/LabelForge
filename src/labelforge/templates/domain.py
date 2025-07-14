"""
Domain-specific labeling function collections.

This module provides pre-built labeling function sets for specific
domains like medical, legal, financial, and others.
"""

from typing import List, Callable, Dict, Any, Optional
from ..types import Example
from .nlp import KeywordLF, RegexLF, SentimentLF


class MedicalLFs:
    """
    Pre-built labeling functions for medical text classification.
    
    Provides LFs for identifying medical content, symptoms, treatments,
    and other healthcare-related categories.
    """
    
    @staticmethod
    def get_symptom_keywords() -> Dict[str, List[str]]:
        """Get symptom keyword collections."""
        return {
            'pain': [
                'pain', 'ache', 'hurt', 'sore', 'tender', 'discomfort',
                'cramping', 'burning', 'stabbing', 'throbbing', 'sharp'
            ],
            'respiratory': [
                'cough', 'wheeze', 'shortness of breath', 'breathing difficulty',
                'chest tightness', 'congestion', 'runny nose', 'sneezing'
            ],
            'gastrointestinal': [
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'bloating',
                'stomach ache', 'heartburn', 'indigestion', 'gas'
            ],
            'neurological': [
                'headache', 'dizziness', 'confusion', 'memory loss', 'seizure',
                'numbness', 'tingling', 'weakness', 'fatigue'
            ]
        }
    
    @staticmethod
    def create_medical_content_lf(name: str = "medical_content") -> Callable[[Example], int]:
        """
        Create LF that identifies medical content.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function that returns 1 for medical content, -1 otherwise
        """
        medical_keywords = {
            # Medical professionals
            'doctor', 'physician', 'nurse', 'surgeon', 'therapist', 'dentist',
            'pharmacist', 'radiologist', 'cardiologist', 'dermatologist',
            
            # Medical facilities
            'hospital', 'clinic', 'emergency room', 'pharmacy', 'laboratory',
            'office', 'medical center', 'urgent care',
            
            # Medical terms
            'patient', 'diagnosis', 'treatment', 'medication', 'prescription',
            'surgery', 'therapy', 'examination', 'test', 'screening',
            'vaccine', 'injection', 'blood pressure', 'heart rate',
            
            # Medical conditions
            'disease', 'illness', 'condition', 'syndrome', 'disorder',
            'infection', 'inflammation', 'allergy', 'chronic', 'acute'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(medical_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )
    
    @staticmethod
    def create_symptom_lf(
        symptom_category: str,
        name: Optional[str] = None
    ) -> Callable[[Example], int]:
        """
        Create LF for specific symptom categories.
        
        Args:
            symptom_category: Category from get_symptom_keywords()
            name: Optional custom name
            
        Returns:
            Labeling function for symptom detection
        """
        symptom_keywords = MedicalLFs.get_symptom_keywords()
        
        if symptom_category not in symptom_keywords:
            raise ValueError(f"Unknown symptom category: {symptom_category}")
        
        lf_name = name or f"{symptom_category}_symptoms"
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=symptom_keywords[symptom_category],
            label=1,
            name=lf_name,
            case_sensitive=False,
            word_boundary=False,  # Allow partial matches for medical terms
            require_all=False
        )
    
    @staticmethod
    def create_medication_mention_lf(name: str = "medication_mention") -> Callable[[Example], int]:
        """
        Create LF that identifies medication mentions.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for medication detection
        """
        # Common medication patterns and keywords
        medication_pattern = r'\\b(?:mg|mcg|ml|tablets?|pills?|capsules?|drops|syrup|cream|ointment)\\b'
        
        return RegexLF.create_regex_lf(
            pattern=medication_pattern,
            label=1,
            name=name
        )


class LegalLFs:
    """
    Pre-built labeling functions for legal text classification.
    
    Provides LFs for identifying legal documents, clauses,
    and legal language patterns.
    """
    
    @staticmethod
    def create_legal_document_lf(name: str = "legal_document") -> Callable[[Example], int]:
        """
        Create LF that identifies legal documents.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for legal document detection
        """
        legal_keywords = {
            # Legal terms
            'contract', 'agreement', 'clause', 'provision', 'statute',
            'regulation', 'compliance', 'liability', 'damages', 'breach',
            'plaintiff', 'defendant', 'litigation', 'lawsuit', 'court',
            
            # Legal phrases
            'whereas', 'therefore', 'hereby', 'pursuant to', 'in accordance with',
            'notwithstanding', 'shall', 'may not', 'is required to',
            
            # Legal entities
            'attorney', 'lawyer', 'counsel', 'judge', 'jury', 'court',
            'tribunal', 'arbitration', 'mediation',
            
            # Legal concepts
            'intellectual property', 'copyright', 'trademark', 'patent',
            'confidentiality', 'non-disclosure', 'indemnification'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(legal_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )
    
    @staticmethod
    def create_contract_clause_lf(name: str = "contract_clause") -> Callable[[Example], int]:
        """
        Create LF that identifies contract clauses.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for contract clause detection
        """
        # Patterns common in contract clauses
        clause_pattern = r'\\b(?:Section|Article|Subsection|Paragraph)\\s+\\d+|\\b(?:shall|must|will|may)\\s+(?:not\\s+)?\\w+'
        
        return RegexLF.create_regex_lf(
            pattern=clause_pattern,
            label=1,
            name=name
        )


class FinancialLFs:
    """
    Pre-built labeling functions for financial text classification.
    
    Provides LFs for identifying financial content, market sentiment,
    and financial entities.
    """
    
    @staticmethod
    def create_financial_content_lf(name: str = "financial_content") -> Callable[[Example], int]:
        """
        Create LF that identifies financial content.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for financial content detection
        """
        financial_keywords = {
            # Financial terms
            'revenue', 'profit', 'loss', 'earnings', 'investment', 'portfolio',
            'dividend', 'stock', 'share', 'bond', 'fund', 'asset', 'liability',
            'equity', 'debt', 'capital', 'market', 'trading', 'financial',
            
            # Financial metrics
            'ROI', 'P/E ratio', 'market cap', 'volatility', 'yield', 'return',
            'growth', 'margin', 'EBITDA', 'cash flow', 'balance sheet',
            
            # Financial entities
            'bank', 'broker', 'exchange', 'SEC', 'Federal Reserve', 'NYSE',
            'NASDAQ', 'mutual fund', 'hedge fund', 'pension fund',
            
            # Currency and amounts
            'dollar', 'USD', 'EUR', 'GBP', 'million', 'billion', 'trillion'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(financial_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )
    
    @staticmethod
    def create_market_sentiment_lf(name: str = "market_sentiment") -> Callable[[Example], int]:
        """
        Create LF for financial market sentiment.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function: 1 for bullish, 0 for bearish, -1 for neutral
        """
        bullish_keywords = {
            'bullish', 'bull market', 'rally', 'surge', 'rise', 'gain',
            'increase', 'growth', 'positive', 'strong', 'outperform',
            'buy', 'upward', 'momentum', 'boom', 'recovery'
        }
        
        bearish_keywords = {
            'bearish', 'bear market', 'decline', 'fall', 'drop', 'loss',
            'decrease', 'negative', 'weak', 'underperform', 'sell',
            'downward', 'crash', 'recession', 'correction'
        }
        
        from ..lf import lf
        
        @lf(name=name)
        def market_sentiment_lf(example: Example) -> int:
            text_lower = example.text.lower()
            
            bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
            
            if bullish_count > bearish_count and bullish_count >= 1:
                return 1  # Bullish
            elif bearish_count > bullish_count and bearish_count >= 1:
                return 0  # Bearish
            else:
                return -1  # Neutral/Abstain
        
        return market_sentiment_lf
    
    @staticmethod
    def create_earnings_report_lf(name: str = "earnings_report") -> Callable[[Example], int]:
        """
        Create LF that identifies earnings reports.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for earnings report detection
        """
        earnings_keywords = {
            'earnings', 'quarterly', 'Q1', 'Q2', 'Q3', 'Q4', 'fiscal year',
            'net income', 'gross profit', 'operating income', 'EPS',
            'guidance', 'outlook', 'beat estimates', 'miss estimates'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(earnings_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )


class EmailLFs:
    """
    Pre-built labeling functions for email classification.
    
    Provides LFs for spam detection, email categories,
    and email sentiment analysis.
    """
    
    @staticmethod
    def create_spam_lf(name: str = "spam_detection") -> Callable[[Example], int]:
        """
        Create LF for spam email detection.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function: 1 for spam, -1 for not spam
        """
        spam_keywords = {
            # Promotional
            'free', 'limited time', 'act now', "don't miss", 'special offer',
            'discount', 'save money', 'lowest price', 'guarantee',
            
            # Urgency
            'urgent', 'immediate', 'expires', 'hurry', 'last chance',
            'only today', 'deadline', 'quick',
            
            # Financial
            'make money', 'earn cash', 'easy money', 'financial freedom',
            'no credit check', 'loan', 'debt', 'refinance',
            
            # Suspicious patterns
            'click here', 'call now', 'order now', 'buy now', 'subscribe',
            'unsubscribe', 'remove', 'winner', 'congratulations',
            
            # Medical/Adult
            'weight loss', 'viagra', 'casino', 'dating', 'adult'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(spam_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=False,
            require_all=False
        )
    
    @staticmethod
    def create_work_email_lf(name: str = "work_email") -> Callable[[Example], int]:
        """
        Create LF for work-related emails.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for work email detection
        """
        work_keywords = {
            'meeting', 'conference', 'project', 'deadline', 'schedule',
            'budget', 'proposal', 'report', 'presentation', 'client',
            'colleague', 'team', 'manager', 'department', 'office',
            'business', 'professional', 'corporate', 'company'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(work_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )


class ProductReviewLFs:
    """
    Pre-built labeling functions for product review classification.
    
    Provides LFs for review sentiment, product categories,
    and review quality assessment.
    """
    
    @staticmethod
    def create_product_review_lf(name: str = "product_review") -> Callable[[Example], int]:
        """
        Create LF that identifies product reviews.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for product review detection
        """
        review_keywords = {
            # Review indicators
            'review', 'rating', 'stars', 'recommend', 'purchase', 'bought',
            'ordered', 'received', 'delivered', 'quality', 'value',
            
            # Product terms
            'product', 'item', 'brand', 'model', 'size', 'color', 'price',
            'shipping', 'packaging', 'description', 'features',
            
            # Experience terms
            'experience', 'satisfied', 'disappointed', 'expected', 'works',
            'broken', 'defective', 'return', 'refund', 'customer service'
        }
        
        return KeywordLF.create_simple_keyword_lf(
            keywords=list(review_keywords),
            label=1,
            name=name,
            case_sensitive=False,
            word_boundary=True,
            require_all=False
        )
    
    @staticmethod
    def create_review_sentiment_lf(name: str = "review_sentiment") -> Callable[[Example], int]:
        """
        Create LF for product review sentiment.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function: 1 for positive, 0 for negative, -1 for neutral
        """
        # Use the SentimentLF but with review-specific keywords
        sentiment_lf = SentimentLF()
        
        # Add review-specific positive keywords
        sentiment_lf.positive_keywords.update({
            'recommend', 'worth it', 'value for money', 'high quality',
            'fast shipping', 'as described', 'exceeded expectations',
            'well made', 'durable', 'stylish', 'comfortable', 'convenient'
        })
        
        # Add review-specific negative keywords
        sentiment_lf.negative_keywords.update({
            'waste of money', 'poor quality', 'cheaply made', 'overpriced',
            'slow shipping', 'not as described', 'fell apart', 'uncomfortable',
            'difficult to use', 'defective', 'returned', 'regret buying'
        })
        
        return sentiment_lf.create_keyword_sentiment_lf(
            name=name,
            positive_label=1,
            negative_label=0,
            threshold=1
        )
    
    @staticmethod
    def create_detailed_review_lf(name: str = "detailed_review") -> Callable[[Example], int]:
        """
        Create LF that identifies detailed/helpful reviews.
        
        Args:
            name: Name for the labeling function
            
        Returns:
            Labeling function for detailed review detection
        """
        from .nlp import LengthLF
        from ..lf import lf
        
        @lf(name=name)
        def detailed_review_lf(example: Example) -> int:
            text = example.text
            
            # Check length (detailed reviews are usually longer)
            word_count = len(text.split())
            if word_count < 20:  # Too short to be detailed
                return -1
            
            # Check for specific details
            detail_indicators = {
                'pros:', 'cons:', 'however', 'although', 'but', 'except',
                'specifically', 'particularly', 'for example', 'such as',
                'compared to', 'in my experience', 'after using', 'update:'
            }
            
            text_lower = text.lower()
            detail_count = sum(1 for indicator in detail_indicators if indicator in text_lower)
            
            # Check for measurements, numbers, specific descriptions
            import re
            has_measurements = bool(re.search(r'\\d+\\s*(?:inch|cm|mm|lb|kg|oz|ml|liter)', text_lower))
            has_time_reference = bool(re.search(r'\\d+\\s*(?:day|week|month|year)', text_lower))
            has_numbers = bool(re.search(r'\\d+', text))
            
            detail_score = detail_count + (1 if has_measurements else 0) + (1 if has_time_reference else 0) + (1 if has_numbers else 0)
            
            if detail_score >= 2 and word_count >= 50:
                return 1  # Detailed review
            else:
                return -1  # Not detailed enough
        
        return detailed_review_lf
