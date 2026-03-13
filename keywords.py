from keybert import KeyBERT
import os

# Global model for lazy loading
_kw_model = None

# Configure cache directory
os.makedirs(os.path.expanduser("~/.cache/huggingface/hub"), exist_ok=True)

def get_keyword_model(timeout=60):
    """Lazily load KeyBERT model on first use.
    
    Args:
        timeout: Network timeout in seconds
    
    Returns:
        KeyBERT instance or None on failure
    """
    global _kw_model
    
    if _kw_model is not None:
        return _kw_model
    
    try:
        print("⏳ Loading KeyBERT model for keyword extraction (first run may take a minute)...")
        _kw_model = KeyBERT()
        print("✅ KeyBERT model loaded successfully")
        return _kw_model
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "connection" in error_msg:
            print("⚠️ Network timeout loading KeyBERT model - keywords unavailable")
            print(f"   Error: {str(e)[:100]}")
        else:
            print(f"⚠️ Error loading KeyBERT model: {str(e)[:100]}")
        return None

def extract_keywords(segment, num_keywords=5):
    """Extract keywords from a segment text.
    
    Args:
        segment: Text to extract keywords from
        num_keywords: Number of keywords to extract (default: 5)
    
    Returns:
        List of keyword strings (guaranteed to be Python strings)
    """
    kw_model = get_keyword_model()
    
    try:
        # Validate input
        if not segment or len(segment.strip()) < 10:
            # Return empty list for short/empty text
            return []
        
        if kw_model is None:
            # Model unavailable - return empty list
            return []
        
        # Extract keywords
        keywords = kw_model.extract_keywords(segment, top_n=num_keywords)
        
        # Convert to list of strings
        result = [str(k[0]) for k in keywords]
        
        return result[:num_keywords]  # Ensure we don't exceed requested count
    
    except Exception as e:
        print(f"Warning: Keyword extraction failed: {str(e)}")
        return []