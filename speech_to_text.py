from faster_whisper import WhisperModel
import math
import os

# WhisperModel configuration
_model = None

# Configure cache directory for model
os.makedirs(os.path.expanduser("~/.cache/whisper"), exist_ok=True)

def get_model(timeout=300):
    """Lazily load WhisperModel on first use.
    
    Args:
        timeout: Network timeout in seconds (longer for large model)
    
    Returns:
        WhisperModel instance or None on failure
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        print("⏳ Loading Whisper model 'base' (first run may take 1-2 minutes)...")
        _model = WhisperModel("base", compute_type="int8")
        print("✅ Whisper model loaded successfully")
        return _model
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "connection" in error_msg:
            print("⚠️ Network timeout loading Whisper model - transcription unavailable")
            print(f"   Error: {str(e)[:100]}")
        else:
            print(f"⚠️ Error loading Whisper model: {str(e)[:100]}")
        return None

def transcribe(audio_path):
    """Transcribe audio and segment it intelligently.
    
    Ensures 7-12 segments per audio for optimal analysis.
    """
    model = get_model()
    
    if model is None:
        print("⚠️ Whisper model not available - returning empty transcription")
        return "", []
    
    segments, info = model.transcribe(audio_path)
    
    transcript = ""
    segment_data = []
    
    for seg in segments:
        transcript += seg.text + " "
        segment_data.append({
            "id": len(segment_data) + 1,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "duration": seg.end - seg.start
        })
    
    # Rebalance segments to 7-12 range if needed
    segment_data = rebalance_segments(segment_data, target_range=(7, 12))
    
    return transcript.strip(), segment_data


def rebalance_segments(segments, target_range=(7, 12)):
    """Rebalance segments to fall within target range.
    
    If too few segments: merge consecutive ones.
    If too many segments: aggregate into fewer groups.
    """
    min_segments, max_segments = target_range
    current_count = len(segments)
    
    if current_count < min_segments:
        # Merge segments: combine every N consecutive segments
        merge_factor = math.ceil(min_segments / current_count)
        segments = merge_consecutive_segments(segments, merge_factor)
    
    elif current_count > max_segments:
        # Group segments: use ceiling division to ensure we get to max_segments or fewer
        keep_factor = math.ceil(current_count / max_segments)
        segments = aggregate_segments(segments, keep_factor)
    
    # Reassign IDs
    for i, seg in enumerate(segments):
        seg["id"] = i + 1
    
    return segments
    
    # Reassign IDs
    for i, seg in enumerate(segments):
        seg["id"] = i + 1
    
    return segments


def merge_consecutive_segments(segments, merge_factor):
    """Merge consecutive segments by merging every merge_factor segments."""
    merged = []
    for i in range(0, len(segments), merge_factor):
        group = segments[i:i + merge_factor]
        if group:
            merged_seg = {
                "id": len(merged) + 1,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join([s["text"] for s in group]),
                "duration": group[-1]["end"] - group[0]["start"]
            }
            merged.append(merged_seg)
    
    return merged if merged else segments


def aggregate_segments(segments, keep_factor):
    """Group segments by keep_factor to reduce total count."""
    aggregated = []
    for i in range(0, len(segments), keep_factor):
        group = segments[i:i + keep_factor]
        if group:
            agg_seg = {
                "id": len(aggregated) + 1,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join([s["text"] for s in group]),
                "duration": group[-1]["end"] - group[0]["start"]
            }
            aggregated.append(agg_seg)
    
    return aggregated