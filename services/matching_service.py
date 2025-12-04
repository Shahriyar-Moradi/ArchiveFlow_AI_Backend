"""
Fuzzy name matching service for property file auto-attachment
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MatchingService:
    """Service for fuzzy matching of client names and property references"""
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a name for comparison with improved handling"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Trim
        normalized = normalized.strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(mr|mrs|ms|dr|prof|eng|eng\.)\.?\s+', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+(jr|sr|ii|iii|iv)$', '', normalized, flags=re.IGNORECASE)
        
        # Handle hyphen variations (Al-Maktoum vs Al Maktoum)
        normalized = re.sub(r'\s*-\s*', ' ', normalized)
        
        # Handle middle name/initial variations
        # "Ahmed Al Maktoum" vs "Ahmed A. Maktoum" vs "A. Al Maktoum"
        # Normalize single letter initials to full form if possible
        normalized = re.sub(r'\b([a-z])\s+', r'\1 ', normalized)  # Keep single letters but normalize spacing
        
        # Remove special characters except spaces and hyphens (already normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return MatchingService.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity score between two strings (0.0 to 1.0)"""
        if not s1 or not s2:
            return 0.0
        
        normalized1 = MatchingService.normalize_name(s1)
        normalized2 = MatchingService.normalize_name(s2)
        
        if normalized1 == normalized2:
            return 1.0
        
        # Check substring match
        if normalized1 in normalized2 or normalized2 in normalized1:
            return 0.85
        
        # Calculate Levenshtein distance
        max_len = max(len(normalized1), len(normalized2))
        if max_len == 0:
            return 1.0
        
        distance = MatchingService.levenshtein_distance(normalized1, normalized2)
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    @staticmethod
    def match_client_name(
        extracted_name: str,
        candidate_names: List[str],
        threshold: float = 0.75
    ) -> List[Tuple[str, float]]:
        """Match extracted client name against candidate names"""
        matches = []
        
        for candidate in candidate_names:
            score = MatchingService.similarity_score(extracted_name, candidate)
            if score >= threshold:
                matches.append((candidate, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    @staticmethod
    def match_property_reference(
        extracted_reference: str,
        candidate_references: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Match extracted property reference against candidate references"""
        matches = []
        
        # Normalize references (case-insensitive, trim)
        normalized_extracted = extracted_reference.lower().strip() if extracted_reference else ""
        
        for candidate in candidate_references:
            normalized_candidate = candidate.lower().strip() if candidate else ""
            
            # Exact match
            if normalized_extracted == normalized_candidate:
                matches.append((candidate, 1.0))
                continue
            
            # Substring match
            if normalized_extracted in normalized_candidate or normalized_candidate in normalized_extracted:
                matches.append((candidate, 0.9))
                continue
            
            # Similarity score
            score = MatchingService.similarity_score(extracted_reference, candidate)
            if score >= threshold:
                matches.append((candidate, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    @staticmethod
    def calculate_confidence(
        name_score: float,
        reference_score: Optional[float] = None
    ) -> float:
        """Calculate overall confidence score for a match"""
        if reference_score is not None:
            # Average of name and reference scores, weighted slightly toward name
            return (name_score * 0.6) + (reference_score * 0.4)
        else:
            # Only name match available
            return name_score * 0.8  # Slightly lower confidence without reference
    
    @staticmethod
    def should_auto_attach(confidence: float, threshold: float = 0.8) -> bool:
        """Determine if auto-attachment should proceed based on confidence"""
        return confidence >= threshold
    
    @staticmethod
    def needs_review(confidence: float, low_threshold: float = 0.6, high_threshold: float = 0.8) -> bool:
        """Determine if match needs manual review"""
        return low_threshold <= confidence < high_threshold
    
    @staticmethod
    def match_multiple_documents(documents: List[Dict[str, Any]], threshold: float = 0.75) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group multiple documents by likely client matches.
        
        Args:
            documents: List of document dicts with 'client_full_name_extracted' field
            threshold: Minimum similarity score to consider documents as matching
            
        Returns:
            Dict mapping normalized client names to lists of matching documents
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        processed_docs = []
        
        for doc in documents:
            client_name = doc.get('client_full_name_extracted') or doc.get('client_full_name', '')
            if not client_name:
                continue
            
            normalized_name = MatchingService.normalize_name(client_name)
            
            # Try to find existing group with similar name
            matched_group = None
            best_score = 0.0
            
            for group_name, group_docs in groups.items():
                score = MatchingService.similarity_score(normalized_name, group_name)
                if score >= threshold and score > best_score:
                    best_score = score
                    matched_group = group_name
            
            if matched_group:
                # Add to existing group
                groups[matched_group].append(doc)
            else:
                # Create new group
                groups[normalized_name] = [doc]
            
            processed_docs.append(doc)
        
        return groups

