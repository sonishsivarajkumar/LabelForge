"""
Pre-built labeling function templates for common use cases.

This module provides ready-to-use labeling function templates for various
domains and NLP tasks, making it easier to get started with weak supervision.
"""

from .nlp import (
    SentimentLF, TopicLF, NERBasedLF, 
    KeywordLF, RegexLF, LengthLF
)
from .domain import (
    MedicalLFs, LegalLFs, FinancialLFs,
    EmailLFs, ProductReviewLFs
)
from .tools import RegexBuilder, RuleMiner, LFTester

__all__ = [
    # NLP Templates
    "SentimentLF",
    "TopicLF", 
    "NERBasedLF",
    "KeywordLF",
    "RegexLF",
    "LengthLF",
    
    # Domain-specific Templates
    "MedicalLFs",
    "LegalLFs", 
    "FinancialLFs",
    "EmailLFs",
    "ProductReviewLFs",
    
    # Development Tools
    "RegexBuilder",
    "RuleMiner",
    "LFTester",
]

# Create a container class for all domain templates
class DomainTemplates:
    """Container for all domain-specific labeling function templates."""
    Medical = MedicalLFs
    Legal = LegalLFs
    Financial = FinancialLFs
    Email = EmailLFs
    ProductReview = ProductReviewLFs

# Add to exports
__all__.append("DomainTemplates")
