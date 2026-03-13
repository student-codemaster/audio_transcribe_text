from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# T5 model configuration
model_name = "t5-small"

# Global variables for lazy loading
_model = None
_tokenizer = None

# Configure cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir

def get_model(timeout=60):
    """Lazily load T5 model and tokenizer on first use.
    
    Args:
        timeout: Network timeout in seconds
    
    Returns:
        Tuple of (model, tokenizer) or (None, None) on failure
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        print(f"⏳ Loading {model_name} model (this may take a minute on first run)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"✅ {model_name} model loaded successfully")
        return _model, _tokenizer
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "connection" in error_msg:
            print(f"⚠️ Network timeout loading {model_name} - summaries unavailable")
            print(f"   Error: {str(e)[:100]}")
        else:
            print(f"⚠️ Error loading {model_name}: {str(e)[:100]}")
        return None, None

def summarize_segments(segments):
    """Summarize text segments using T5 model.
    
    Args:
        segments: List of text strings to summarize
    
    Returns:
        List of summary strings (or original text if model unavailable)
    """
    model, tokenizer = get_model()
    
    # If model loading failed, return empty summaries as fallback
    if model is None or tokenizer is None:
        print("⚠️ Using empty summaries as fallback (model not available)")
        return ["[Summary unavailable]" for _ in segments]
    
    summaries = []

    for seg in segments:
        try:
            # Prepare the input text for T5
            input_text = "summarize: " + seg[:512]  # T5 expects "summarize:" prefix
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=20,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the summary
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)
        except Exception as e:
            # If individual summary fails, use the original text
            summaries.append(f"[Summary error: {str(e)[:50]}]")

    return summaries