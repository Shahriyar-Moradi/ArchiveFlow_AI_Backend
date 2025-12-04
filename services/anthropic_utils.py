"""
Anthropic API utilities and helper functions
"""
import re

def detect_model_not_found_error(error_message: str, model_id: str) -> str:
    """
    Detect if the error is due to model not found and provide helpful message
    
    Args:
        error_message: Error message from Anthropic API
        model_id: The model ID that was requested
        
    Returns:
        Helpful error message if model not found, empty string otherwise
    """
    error_lower = error_message.lower()
    
    # Check for model not found errors
    if any(phrase in error_lower for phrase in [
        'model not found',
        'invalid model',
        'model_not_found',
        'unknown model',
        'model does not exist'
    ]):
        return (
            f"The model '{model_id}' is not available. "
            "Please check your Anthropic API key has access to this model, "
            "or update ANTHROPIC_MODEL in your .env file to use an available model "
            "(e.g., 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229')."
        )
    
    return ""


def extract_model_from_error(error_message: str) -> str:
    """
    Try to extract suggested model from error message
    
    Args:
        error_message: Error message from Anthropic API
        
    Returns:
        Suggested model ID if found, empty string otherwise
    """
    # Try to find model suggestions in error message
    # Pattern: claude-X-Y-YYYYMMDD or claude-X-Y-date
    pattern = r'claude-[\w-]+-\d{8}'
    match = re.search(pattern, error_message)
    
    if match:
        return match.group(0)
    
    return ""

