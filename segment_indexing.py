"""Segment indexing and storage with embeddings."""

from embedding_model import encode_sentences
import json
from datetime import datetime
import numpy as np
import os

class SegmentIndex:
    """Store and index segments with embeddings and metadata."""
    
    def __init__(self):
        self.segments = []
        self.embeddings = []
        self.created_at = datetime.now().isoformat()
    
    def add_segment(self, segment_id, start, end, text, keywords, summary, sentiment_score):
        """Add a segment with all metadata."""
        segment = {
            "summary": summary,
            "segments": text,
            "id": segment_id,
            "start": start,
            "end": end,
            "keywords": keywords,
            "sentiment_score": sentiment_score,
            "duration": end - start,
            "embedding": None  # Will be populated
        }
        self.segments.append(segment)
        return segment
    
    def compute_embeddings(self):
        """Compute embeddings for all segment texts."""
        texts = [seg.get("segments", seg.get("text", "")) for seg in self.segments]
        embeddings = encode_sentences(texts)
        
        for i, seg in enumerate(self.segments):
            # Convert numpy array to list for JSON serialization
            if isinstance(embeddings[i], np.ndarray):
                seg["embedding"] = embeddings[i].tolist()
            else:
                seg["embedding"] = list(embeddings[i])
        
        self.embeddings = embeddings
        return embeddings
    
    def to_dict(self):
        """Convert index to dictionary for serialization."""
        # Convert numpy types to native Python types for JSON compatibility
        segments_clean = []
        for seg in self.segments:
            seg_clean = seg.copy()
            # Convert numpy scalars to Python types
            if isinstance(seg_clean.get("sentiment_score"), np.ndarray):
                seg_clean["sentiment_score"] = float(seg_clean["sentiment_score"])
            elif hasattr(seg_clean.get("sentiment_score"), "item"):
                seg_clean["sentiment_score"] = seg_clean["sentiment_score"].item()
            
            seg_clean["start"] = float(seg_clean["start"])
            seg_clean["end"] = float(seg_clean["end"])
            seg_clean["duration"] = float(seg_clean["duration"])
            segments_clean.append(seg_clean)
        
        return {
            "created_at": self.created_at,
            "total_segments": len(self.segments),
            "segments": segments_clean
        }
    
    def save_to_json(self, filepath):
        """Save segment index to JSON file."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Saved: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving {filepath}: {str(e)}")
            return False
    
    def get_segment_by_id(self, segment_id):
        """Retrieve a segment by ID."""
        for seg in self.segments:
            if seg["id"] == segment_id:
                return seg
        return None
    
    def get_segments_by_time_range(self, start_time, end_time):
        """Get segments within a time range."""
        return [seg for seg in self.segments 
                if seg["start"] >= start_time and seg["end"] <= end_time]
    
    def format_for_display(self):
        """Format segments for UI display."""
        display_data = []
        for seg in self.segments:
            display_data.append({
                "summary": seg["summary"],
                "segments": seg.get("segments", seg.get("text", "")),
                "id": seg["id"],
                "timestamp": f"{self._format_time(seg['start'])} - {self._format_time(seg['end'])}",
                "duration": f"{seg['duration']:.1f}s",
                "keywords": ", ".join(seg["keywords"]),
                "sentiment": seg["sentiment_score"]
            })
        return display_data
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds as HH:MM:SS."""
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"
