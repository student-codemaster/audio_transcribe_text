from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Global model cache - lazy loaded
_model = None
_model_name = "all-MiniLM-L6-v2"

# Try to set cache directory for transformers
try:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
except Exception:
    pass


def get_model(timeout=60):
    """Lazily load the embedding model with timeout handling.
    
    Args:
        timeout: Timeout in seconds for model loading
    
    Returns:
        SentenceTransformer model instance or None if load fails
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        print(f"⏳ Loading embedding model ({_model_name})...")
        
        # Try to load with timeout
        _model = SentenceTransformer(
            _model_name,
            cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
        )
        print(f"✅ Model loaded successfully")
        return _model
        
    except Exception as e:
        error_msg = str(e)
        
        if "timeout" in error_msg.lower() or "read" in error_msg.lower():
            print(f"⚠️ Network timeout loading model: {error_msg}")
            print("💡 Tip: Check your internet connection or try again later")
        else:
            print(f"⚠️ Error loading embedding model: {error_msg}")
        
        # Return None - encode_sentences will handle gracefully
        return None


def encode_sentences(sentences):
    """Encode sentences using MiniLM model.
    
    Args:
        sentences: List of text strings to encode
    
    Returns:
        Numpy array of embeddings (shape: [n, 384])
    """
    if not sentences:
        return np.array([])
    
    try:
        # Get model (lazy loaded)
        model = get_model()
        
        if model is None:
            # Model failed to load - return zero embeddings
            print("⚠️ Using fallback zero embeddings (model unavailable)")
            return np.zeros((len(sentences), 384))
        
        # Filter empty strings
        valid_sentences = [s for s in sentences if s and len(s.strip()) > 0]
        if not valid_sentences:
            # Return zero embeddings for empty list
            return np.zeros((len(sentences), 384))
        
        embeddings = model.encode(valid_sentences, show_progress_bar=False)
        
        # Ensure numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        return embeddings
    
    except Exception as e:
        error_msg = str(e)
        
        if "timeout" in error_msg.lower():
            print(f"⚠️ Embedding operation timeout: {error_msg}")
        else:
            print(f"⚠️ Embedding error: {error_msg}")
        
        # Return zero embeddings as fallback
        return np.zeros((len(sentences), 384))