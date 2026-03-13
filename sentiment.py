from transformers import pipeline
import numpy as np
import os

# Global model for lazy loading
_sentiment_model = None

# Configure cache directory
os.makedirs(os.path.expanduser("~/.cache/huggingface/hub"), exist_ok=True)

def get_sentiment_model(timeout=60):
    """Lazily load sentiment analysis pipeline on first use.
    
    Args:
        timeout: Network timeout in seconds
    
    Returns:
        Sentiment pipeline or None on failure
    """
    global _sentiment_model
    
    if _sentiment_model is not None:
        return _sentiment_model
    
    try:
        print("⏳ Loading sentiment analysis model (first run may take a minute)...")
        _sentiment_model = pipeline("sentiment-analysis")
        print("✅ Sentiment analysis model loaded successfully")
        return _sentiment_model
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "connection" in error_msg:
            print("⚠️ Network timeout loading sentiment model - sentiment scores unavailable")
            print(f"   Error: {str(e)[:100]}")
        else:
            print(f"⚠️ Error loading sentiment model: {str(e)[:100]}")
        return None


def avg_sentiment(segments):
    """Compute sentiment scores for each segment and return average.

    Returns a tuple (average_score, scores_list).
    """
    sentiment_model = get_sentiment_model()
    
    scores = []
    for s in segments:
        if not s or len(s.strip()) == 0:
            scores.append(0.0)
            continue
        
        try:
            if sentiment_model is None:
                scores.append(0.0)
                continue
                
            result = sentiment_model(s[:512])[0]
            score = float(result["score"])  # Ensure it's a Python float
            if result["label"] == "NEGATIVE":
                score = -score
            scores.append(score)
        except Exception as e:
            print(f"Warning: Sentiment analysis failed for segment: {str(e)}")
            scores.append(0.0)
    
    avg = float(sum(scores) / len(scores)) if scores else 0.0
    return avg, [float(s) for s in scores]  # Ensure all scores are Python floats