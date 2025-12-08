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
        """Normalize a name for comparison with improved handling of variations"""
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
        
        # Handle common Arabic/English name abbreviations
        # Map common abbreviations to full forms for better matching
        abbreviation_map = {
            r'\bmohd\b': 'mohammed',
            r'\bmoh\b': 'mohammed',
            r'\bmd\b': 'mohammed',
            r'\bali\b': 'ali',  # Keep as is, but normalize variations
            r'\bala\b': 'ala',
            r'\balaa\b': 'ala',
        }
        for pattern, replacement in abbreviation_map.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Handle middle name/initial variations
        # "Ahmed Al Maktoum" vs "Ahmed A. Maktoum" vs "A. Al Maktoum"
        # Normalize single letter initials - remove periods and normalize spacing
        normalized = re.sub(r'\b([a-z])\.\s*', r'\1 ', normalized)  # Remove period after single letter
        normalized = re.sub(r'\b([a-z])\s+', r'\1 ', normalized)  # Normalize spacing after single letter
        
        # Remove special characters except spaces (hyphens already normalized to spaces)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Final whitespace cleanup
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
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
        """Calculate similarity score between two strings (0.0 to 1.0) with enhanced matching"""
        if not s1 or not s2:
            return 0.0
        
        normalized1 = MatchingService.normalize_name(s1)
        normalized2 = MatchingService.normalize_name(s2)
        
        if normalized1 == normalized2:
            return 1.0
        
        # Check substring match (one name contains the other)
        if normalized1 in normalized2 or normalized2 in normalized1:
            # If one is a substring of the other, check if it's a meaningful match
            # (e.g., "Ahmed Al Maktoum" contains "Ahmed Maktoum")
            shorter = normalized1 if len(normalized1) < len(normalized2) else normalized2
            longer = normalized2 if len(normalized1) < len(normalized2) else normalized1
            
            # Split into words to check if key parts match
            shorter_words = set(shorter.split())
            longer_words = set(longer.split())
            
            # If shorter name's words are mostly in longer name, it's a good match
            if len(shorter_words) > 0:
                overlap_ratio = len(shorter_words & longer_words) / len(shorter_words)
                if overlap_ratio >= 0.8:  # 80% of words match
                    return 0.90
                elif overlap_ratio >= 0.6:  # 60% of words match
                    return 0.80
                else:
                    return 0.75  # Basic substring match
        
        # Check for partial name matches (first name + last name when middle differs)
        words1 = normalized1.split()
        words2 = normalized2.split()
        
        if len(words1) >= 2 and len(words2) >= 2:
            # Check if first and last names match (ignoring middle names)
            first_last_match = (
                words1[0] == words2[0] and  # First name matches
                words1[-1] == words2[-1]   # Last name matches
            )
            
            if first_last_match:
                # Calculate similarity based on how many words match
                set1 = set(words1)
                set2 = set(words2)
                common_words = len(set1 & set2)
                total_unique_words = len(set1 | set2)
                
                if total_unique_words > 0:
                    word_similarity = common_words / total_unique_words
                    # Boost score for first+last match, but reduce slightly for middle name differences
                    base_score = 0.75 + (word_similarity * 0.20)
                    return min(base_score, 0.95)  # Cap at 0.95 for partial matches
        
        # Calculate Levenshtein distance for overall similarity
        max_len = max(len(normalized1), len(normalized2))
        if max_len == 0:
            return 1.0
        
        distance = MatchingService.levenshtein_distance(normalized1, normalized2)
        similarity = 1.0 - (distance / max_len)
        
        # Boost similarity if word sets have significant overlap
        words1_set = set(normalized1.split())
        words2_set = set(normalized2.split())
        if len(words1_set) > 0 and len(words2_set) > 0:
            word_overlap = len(words1_set & words2_set) / len(words1_set | words2_set)
            # Combine Levenshtein similarity with word overlap
            similarity = max(similarity, word_overlap * 0.9)
        
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

