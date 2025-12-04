"""
JSON parsing utilities for OCR response extraction
"""
import json
import re
from typing import Optional, Dict, Any

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain additional commentary
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    if not text:
        return None
    
    # Try direct JSON parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks (```json ... ```)
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text (looking for { ... })
    # Find the first { and last } that form a valid JSON
    brace_start = text.find('{')
    if brace_start != -1:
        # Find matching closing brace
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching brace
                    potential_json = text[brace_start:i+1]
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Try to find JSON array
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(array_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def clean_json_string(text: str) -> str:
    """
    Clean JSON string by removing common issues
    
    Args:
        text: JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (but not inside strings)
    # This is a simplified version - be careful with complex cases
    text = text.replace("'", '"')
    
    return text

