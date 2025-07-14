"""
Development tools for labeling functions.

This module provides utilities for building, testing, and debugging
labeling functions including regex builders, rule miners, and testing frameworks.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Callable, Tuple, Union
from collections import Counter, defaultdict
import warnings

from ..types import Example, LFOutput
from ..lf import get_registered_lfs


class RegexBuilder:
    """
    Interactive regex pattern builder for labeling functions.
    
    Helps users build and test regular expression patterns
    for text classification tasks.
    """
    
    def __init__(self):
        """Initialize regex builder."""
        self.patterns = {}
        self.test_examples = []
    
    def add_pattern(self, name: str, pattern: str, description: str = "") -> None:
        """
        Add a regex pattern to the builder.
        
        Args:
            name: Name for the pattern
            pattern: Regular expression pattern
            description: Optional description of the pattern
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self.patterns[name] = {
                'pattern': pattern,
                'compiled': compiled,
                'description': description,
                'matches': []
            }
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    def test_pattern(
        self, 
        pattern_name: str, 
        examples: List[Example]
    ) -> Dict[str, Any]:
        """
        Test a pattern against examples.
        
        Args:
            pattern_name: Name of pattern to test
            examples: List of examples to test against
            
        Returns:
            Dictionary with test results
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not found")
        
        pattern_info = self.patterns[pattern_name]
        compiled_pattern = pattern_info['compiled']
        
        matches = []
        for i, example in enumerate(examples):
            found_matches = compiled_pattern.findall(example.text)
            if found_matches:
                matches.append({
                    'example_index': i,
                    'text': example.text,
                    'matches': found_matches,
                    'match_count': len(found_matches)
                })
        
        # Store results
        self.patterns[pattern_name]['matches'] = matches
        
        return {
            'pattern_name': pattern_name,
            'pattern': pattern_info['pattern'],
            'total_examples': len(examples),
            'matching_examples': len(matches),
            'match_rate': len(matches) / len(examples) if examples else 0,
            'total_matches': sum(m['match_count'] for m in matches),
            'matches': matches
        }
    
    def suggest_improvements(
        self, 
        pattern_name: str, 
        target_examples: List[Example],
        non_target_examples: List[Example]
    ) -> List[str]:
        """
        Suggest improvements for a pattern based on target/non-target examples.
        
        Args:
            pattern_name: Name of pattern to improve
            target_examples: Examples that should match
            non_target_examples: Examples that should not match
            
        Returns:
            List of improvement suggestions
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not found")
        
        compiled_pattern = self.patterns[pattern_name]['compiled']
        suggestions = []
        
        # Check false negatives (target examples that don't match)
        false_negatives = []
        for example in target_examples:
            if not compiled_pattern.search(example.text):
                false_negatives.append(example.text)
        
        if false_negatives:
            suggestions.append(f"Pattern misses {len(false_negatives)} target examples")
            
            # Analyze common words in false negatives
            words = []
            for text in false_negatives[:5]:  # Analyze first 5
                words.extend(text.lower().split())
            
            common_words = Counter(words).most_common(5)
            if common_words:
                suggestions.append(f"Consider adding patterns for: {[word for word, count in common_words]}")
        
        # Check false positives (non-target examples that match)
        false_positives = []
        for example in non_target_examples:
            if compiled_pattern.search(example.text):
                false_positives.append(example.text)
        
        if false_positives:
            suggestions.append(f"Pattern incorrectly matches {len(false_positives)} non-target examples")
            suggestions.append("Consider making pattern more specific or adding negative lookaheads")
        
        return suggestions
    
    def export_pattern_as_lf(
        self, 
        pattern_name: str, 
        label: int,
        lf_name: Optional[str] = None
    ) -> str:
        """
        Export a pattern as labeling function code.
        
        Args:
            pattern_name: Name of pattern to export
            label: Label to assign for matches
            lf_name: Optional name for the LF
            
        Returns:
            Python code string for the labeling function
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not found")
        
        pattern_info = self.patterns[pattern_name]
        pattern = pattern_info['pattern']
        description = pattern_info['description']
        
        lf_name = lf_name or f"{pattern_name}_lf"
        
        code = f'''
from labelforge.templates.nlp import RegexLF

# {description}
{lf_name} = RegexLF.create_regex_lf(
    pattern=r"{pattern}",
    label={label},
    name="{lf_name}"
)
'''
        return code.strip()


class RuleMiner:
    """
    Automatic rule mining for labeling function discovery.
    
    Analyzes text data to suggest potential labeling function rules
    based on patterns, keywords, and statistical analysis.
    """
    
    def __init__(self):
        """Initialize rule miner."""
        self.analyzed_examples = []
        self.discovered_rules = []
    
    def mine_keyword_rules(
        self,
        positive_examples: List[Example],
        negative_examples: List[Example],
        min_frequency: int = 3,
        max_keywords: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Mine keyword-based rules from examples.
        
        Args:
            positive_examples: Examples with positive label
            negative_examples: Examples with negative label
            min_frequency: Minimum frequency for keyword consideration
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of discovered keyword rules
        """
        # Extract words from positive and negative examples
        pos_words = []
        neg_words = []
        
        for example in positive_examples:
            words = re.findall(r'\\b\\w+\\b', example.text.lower())
            pos_words.extend(words)
        
        for example in negative_examples:
            words = re.findall(r'\\b\\w+\\b', example.text.lower())
            neg_words.extend(words)
        
        # Count word frequencies
        pos_counter = Counter(pos_words)
        neg_counter = Counter(neg_words)
        
        # Find discriminative keywords
        keyword_rules = []
        
        for word, pos_count in pos_counter.most_common():
            if pos_count < min_frequency:
                break
            
            neg_count = neg_counter.get(word, 0)
            
            # Calculate discrimination score
            total_pos = len(positive_examples)
            total_neg = len(negative_examples)
            
            pos_rate = pos_count / total_pos if total_pos > 0 else 0
            neg_rate = neg_count / total_neg if total_neg > 0 else 0
            
            # Avoid division by zero
            if neg_rate == 0:
                discrimination_score = pos_rate
            else:
                discrimination_score = pos_rate / (pos_rate + neg_rate)
            
            if discrimination_score > 0.7:  # Strong positive indicator
                keyword_rules.append({
                    'type': 'keyword',
                    'keyword': word,
                    'pos_frequency': pos_count,
                    'neg_frequency': neg_count,
                    'pos_rate': pos_rate,
                    'neg_rate': neg_rate,
                    'discrimination_score': discrimination_score,
                    'suggested_label': 1
                })
        
        # Sort by discrimination score and limit
        keyword_rules.sort(key=lambda x: x['discrimination_score'], reverse=True)
        return keyword_rules[:max_keywords]
    
    def mine_length_rules(
        self,
        examples_with_labels: List[Tuple[Example, int]],
        min_examples_per_bin: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Mine length-based rules from examples.
        
        Args:
            examples_with_labels: List of (example, label) tuples
            min_examples_per_bin: Minimum examples per length bin
            
        Returns:
            List of discovered length rules
        """
        # Group examples by label and calculate length statistics
        label_lengths = defaultdict(list)
        
        for example, label in examples_with_labels:
            char_length = len(example.text)
            word_length = len(example.text.split())
            
            label_lengths[label].append({
                'char_length': char_length,
                'word_length': word_length
            })
        
        length_rules = []
        
        for label, lengths in label_lengths.items():
            if len(lengths) < min_examples_per_bin:
                continue
            
            char_lengths = [l['char_length'] for l in lengths]
            word_lengths = [l['word_length'] for l in lengths]
            
            char_mean = np.mean(char_lengths)
            char_std = np.std(char_lengths)
            word_mean = np.mean(word_lengths)
            word_std = np.std(word_lengths)
            
            # Suggest rules based on statistical analysis
            if char_std < char_mean * 0.3:  # Low variance suggests consistent length
                length_rules.append({
                    'type': 'char_length',
                    'label': label,
                    'min_length': int(char_mean - char_std),
                    'max_length': int(char_mean + char_std),
                    'mean_length': char_mean,
                    'confidence': 1 - (char_std / char_mean) if char_mean > 0 else 0
                })
            
            if word_std < word_mean * 0.3:  # Low variance suggests consistent word count
                length_rules.append({
                    'type': 'word_length',
                    'label': label,
                    'min_words': int(word_mean - word_std),
                    'max_words': int(word_mean + word_std),
                    'mean_words': word_mean,
                    'confidence': 1 - (word_std / word_mean) if word_mean > 0 else 0
                })
        
        return length_rules
    
    def mine_pattern_rules(
        self,
        positive_examples: List[Example],
        negative_examples: List[Example],
        min_pattern_length: int = 3,
        max_patterns: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Mine regex pattern rules from examples.
        
        Args:
            positive_examples: Examples with positive label
            negative_examples: Examples with negative label
            min_pattern_length: Minimum pattern length to consider
            max_patterns: Maximum number of patterns to return
            
        Returns:
            List of discovered pattern rules
        """
        # Extract common character patterns from positive examples
        pos_texts = [example.text for example in positive_examples]
        neg_texts = [example.text for example in negative_examples]
        
        pattern_rules = []
        
        # Look for common email patterns
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        pos_email_matches = sum(1 for text in pos_texts if re.search(email_pattern, text))
        neg_email_matches = sum(1 for text in neg_texts if re.search(email_pattern, text))
        
        if pos_email_matches > len(pos_texts) * 0.3:  # At least 30% contain emails
            pattern_rules.append({
                'type': 'regex_pattern',
                'pattern': email_pattern,
                'description': 'Email address pattern',
                'pos_matches': pos_email_matches,
                'neg_matches': neg_email_matches,
                'precision': pos_email_matches / (pos_email_matches + neg_email_matches) if (pos_email_matches + neg_email_matches) > 0 else 0
            })
        
        # Look for URL patterns
        url_pattern = r'https?://(?:[-\\w.])+(?:[:\\d]+)?(?:/(?:[\\w/_.])*(?:\\?(?:[\\w&=%.])*)?(?:#(?:[\\w.])*)?)?'
        pos_url_matches = sum(1 for text in pos_texts if re.search(url_pattern, text))
        neg_url_matches = sum(1 for text in neg_texts if re.search(url_pattern, text))
        
        if pos_url_matches > len(pos_texts) * 0.3:
            pattern_rules.append({
                'type': 'regex_pattern',
                'pattern': url_pattern,
                'description': 'URL pattern',
                'pos_matches': pos_url_matches,
                'neg_matches': neg_url_matches,
                'precision': pos_url_matches / (pos_url_matches + neg_url_matches) if (pos_url_matches + neg_url_matches) > 0 else 0
            })
        
        # Look for phone number patterns
        phone_pattern = r'\\b(?:\\+?1[-.]?)?\\(?([0-9]{3})\\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\\b'
        pos_phone_matches = sum(1 for text in pos_texts if re.search(phone_pattern, text))
        neg_phone_matches = sum(1 for text in neg_texts if re.search(phone_pattern, text))
        
        if pos_phone_matches > len(pos_texts) * 0.3:
            pattern_rules.append({
                'type': 'regex_pattern',
                'pattern': phone_pattern,
                'description': 'Phone number pattern',
                'pos_matches': pos_phone_matches,
                'neg_matches': neg_phone_matches,
                'precision': pos_phone_matches / (pos_phone_matches + neg_phone_matches) if (pos_phone_matches + neg_phone_matches) > 0 else 0
            })
        
        return pattern_rules[:max_patterns]
    
    def generate_lf_code(self, rule: Dict[str, Any], lf_name: str) -> str:
        """
        Generate LF code from a discovered rule.
        
        Args:
            rule: Rule dictionary from mining methods
            lf_name: Name for the generated LF
            
        Returns:
            Python code string for the labeling function
        """
        rule_type = rule['type']
        
        if rule_type == 'keyword':
            return f'''
from labelforge.templates.nlp import KeywordLF

{lf_name} = KeywordLF.create_simple_keyword_lf(
    keywords=["{rule['keyword']}"],
    label={rule['suggested_label']},
    name="{lf_name}",
    case_sensitive=False,
    word_boundary=True
)
'''
        
        elif rule_type == 'char_length':
            return f'''
from labelforge.templates.nlp import LengthLF

{lf_name} = LengthLF.create_char_length_lf(
    min_length={rule['min_length']},
    max_length={rule['max_length']},
    label={rule['label']},
    name="{lf_name}"
)
'''
        
        elif rule_type == 'word_length':
            return f'''
from labelforge.templates.nlp import LengthLF

{lf_name} = LengthLF.create_word_count_lf(
    min_words={rule['min_words']},
    max_words={rule['max_words']},
    label={rule['label']},
    name="{lf_name}"
)
'''
        
        elif rule_type == 'regex_pattern':
            return f'''
from labelforge.templates.nlp import RegexLF

# {rule['description']}
{lf_name} = RegexLF.create_regex_lf(
    pattern=r"{rule['pattern']}",
    label=1,
    name="{lf_name}"
)
'''
        
        else:
            return f"# Unknown rule type: {rule_type}"


class LFTester:
    """
    Testing framework for labeling functions.
    
    Provides utilities for testing labeling function performance,
    coverage, conflicts, and other quality metrics.
    """
    
    def __init__(self):
        """Initialize LF tester."""
        self.test_results = {}
    
    def test_coverage(
        self,
        lf_output: LFOutput,
        examples: List[Example]
    ) -> Dict[str, Any]:
        """
        Test labeling function coverage.
        
        Args:
            lf_output: LFOutput from applying LFs
            examples: List of examples
            
        Returns:
            Coverage analysis results
        """
        votes = lf_output.votes
        n_examples, n_lfs = votes.shape
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        # Calculate per-LF coverage
        lf_coverage = []
        for i in range(n_lfs):
            coverage = np.sum(votes[:, i] != -1) / n_examples
            lf_coverage.append({
                'lf_name': lf_names[i],
                'coverage': coverage,
                'abstentions': np.sum(votes[:, i] == -1),
                'votes': np.sum(votes[:, i] != -1)
            })
        
        # Overall coverage
        total_coverage = np.mean(np.any(votes != -1, axis=1))
        
        # Coverage overlap analysis
        overlap_matrix = np.zeros((n_lfs, n_lfs))
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i != j:
                    both_vote = np.sum((votes[:, i] != -1) & (votes[:, j] != -1))
                    overlap_matrix[i, j] = both_vote / n_examples
        
        return {
            'total_coverage': total_coverage,
            'lf_coverage': lf_coverage,
            'overlap_matrix': overlap_matrix,
            'uncovered_examples': np.sum(~np.any(votes != -1, axis=1)),
            'mean_lf_coverage': np.mean([lf['coverage'] for lf in lf_coverage])
        }
    
    def test_conflicts(
        self,
        lf_output: LFOutput,
        examples: List[Example]
    ) -> Dict[str, Any]:
        """
        Test labeling function conflicts.
        
        Args:
            lf_output: LFOutput from applying LFs
            examples: List of examples
            
        Returns:
            Conflict analysis results
        """
        votes = lf_output.votes
        n_examples, n_lfs = votes.shape
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        # Find conflicting examples
        conflicts = []
        conflict_count = 0
        
        for i in range(n_examples):
            example_votes = votes[i][votes[i] != -1]
            if len(example_votes) > 1 and len(np.unique(example_votes)) > 1:
                conflict_count += 1
                
                # Find which LFs are conflicting
                conflicting_lfs = []
                for j in range(n_lfs):
                    if votes[i, j] != -1:
                        conflicting_lfs.append((lf_names[j], int(votes[i, j])))
                
                conflicts.append({
                    'example_index': i,
                    'text': examples[i].text,
                    'conflicting_lfs': conflicting_lfs,
                    'unique_votes': len(np.unique(example_votes))
                })
        
        # Pairwise conflict analysis
        pairwise_conflicts = []
        for i in range(n_lfs):
            for j in range(i + 1, n_lfs):
                both_vote = (votes[:, i] != -1) & (votes[:, j] != -1)
                if np.sum(both_vote) > 0:
                    disagreements = votes[both_vote, i] != votes[both_vote, j]
                    conflict_rate = np.mean(disagreements)
                    
                    pairwise_conflicts.append({
                        'lf1': lf_names[i],
                        'lf2': lf_names[j],
                        'overlap_count': np.sum(both_vote),
                        'conflict_count': np.sum(disagreements),
                        'conflict_rate': conflict_rate
                    })
        
        return {
            'total_conflicts': conflict_count,
            'conflict_rate': conflict_count / n_examples,
            'conflicts': conflicts[:10],  # First 10 conflicts
            'pairwise_conflicts': pairwise_conflicts
        }
    
    def test_individual_lf(
        self,
        lf_func: Callable[[Example], int],
        examples: List[Example],
        true_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Test an individual labeling function.
        
        Args:
            lf_func: Labeling function to test
            examples: List of examples
            true_labels: Optional true labels for supervised evaluation
            
        Returns:
            Individual LF test results
        """
        # Apply LF to examples
        votes = []
        errors = []
        
        for i, example in enumerate(examples):
            try:
                vote = lf_func(example)
                votes.append(vote)
            except Exception as e:
                votes.append(-1)  # Abstain on error
                errors.append({
                    'example_index': i,
                    'error': str(e),
                    'text': example.text[:100]  # First 100 chars
                })
        
        votes = np.array(votes)
        
        # Basic statistics
        coverage = np.sum(votes != -1) / len(votes)
        abstentions = np.sum(votes == -1)
        
        # Vote distribution
        unique_votes, vote_counts = np.unique(votes[votes != -1], return_counts=True)
        vote_distribution = dict(zip(unique_votes.astype(int), vote_counts))
        
        results = {
            'coverage': coverage,
            'abstentions': abstentions,
            'vote_distribution': vote_distribution,
            'error_count': len(errors),
            'errors': errors[:5]  # First 5 errors
        }
        
        # Supervised evaluation if true labels provided
        if true_labels is not None:
            true_labels = np.array(true_labels)
            
            # Only evaluate non-abstaining predictions
            non_abstain_mask = votes != -1
            if np.sum(non_abstain_mask) > 0:
                pred_votes = votes[non_abstain_mask]
                pred_labels = true_labels[non_abstain_mask]
                
                accuracy = np.mean(pred_votes == pred_labels)
                
                # Calculate precision and recall for each class
                class_metrics = {}
                for vote_class in unique_votes:
                    if vote_class != -1:
                        vote_class = int(vote_class)
                        tp = np.sum((pred_votes == vote_class) & (pred_labels == vote_class))
                        fp = np.sum((pred_votes == vote_class) & (pred_labels != vote_class))
                        fn = np.sum((pred_votes != vote_class) & (pred_labels == vote_class))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        class_metrics[vote_class] = {
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'support': np.sum(pred_labels == vote_class)
                        }
                
                results['supervised_metrics'] = {
                    'accuracy': accuracy,
                    'class_metrics': class_metrics
                }
        
        return results
    
    def generate_test_report(
        self,
        lf_output: LFOutput,
        examples: List[Example],
        true_labels: Optional[List[int]] = None
    ) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            lf_output: LFOutput from applying LFs
            examples: List of examples
            true_labels: Optional true labels
            
        Returns:
            Formatted test report as string
        """
        coverage_results = self.test_coverage(lf_output, examples)
        conflict_results = self.test_conflicts(lf_output, examples)
        
        report_lines = [
            "# Labeling Function Test Report",
            "=" * 40,
            "",
            "## Coverage Analysis",
            f"• Total coverage: {coverage_results['total_coverage']:.1%}",
            f"• Mean LF coverage: {coverage_results['mean_lf_coverage']:.1%}",
            f"• Uncovered examples: {coverage_results['uncovered_examples']}",
            "",
            "### Per-LF Coverage:",
        ]
        
        for lf_info in coverage_results['lf_coverage']:
            report_lines.append(f"• {lf_info['lf_name']}: {lf_info['coverage']:.1%} ({lf_info['votes']} votes)")
        
        report_lines.extend([
            "",
            "## Conflict Analysis",
            f"• Total conflicts: {conflict_results['total_conflicts']}",
            f"• Conflict rate: {conflict_results['conflict_rate']:.1%}",
            "",
        ])
        
        if conflict_results['pairwise_conflicts']:
            report_lines.append("### Top Pairwise Conflicts:")
            sorted_conflicts = sorted(
                conflict_results['pairwise_conflicts'],
                key=lambda x: x['conflict_rate'],
                reverse=True
            )
            
            for conflict in sorted_conflicts[:5]:
                report_lines.append(
                    f"• {conflict['lf1']} vs {conflict['lf2']}: "
                    f"{conflict['conflict_rate']:.1%} ({conflict['conflict_count']}/{conflict['overlap_count']})"
                )
        
        return "\\n".join(report_lines)
