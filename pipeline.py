from audio_preprocess import preprocess_audio
from speech_to_text import transcribe
from keywords import extract_keywords
from sentiment import avg_sentiment
from summarizer import summarize_segments
from segment_indexing import SegmentIndex
from search import SegmentSearch
import os
import json
import shutil

def run_pipeline(audio_path, save_output=True, output_dir="final_outputs"):
    """Run the complete podcast analysis pipeline.
    
    Args:
        audio_path: Path to audio file
        save_output: Whether to save results to JSON
        output_dir: Directory to save outputs
    
    Returns: (transcript, topics, sentiment_score, summaries, segment_index, search_engine)
    """
    try:
        # ensure audio gets stored in data folder
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        abs_audio = os.path.abspath(audio_path)
        if not abs_audio.startswith(os.path.abspath(data_dir) + os.sep):
            dest = os.path.join(data_dir, os.path.basename(audio_path))
            counter = 1
            base, ext = os.path.splitext(dest)
            while os.path.exists(dest):
                dest = f"{base}_{counter}{ext}"
                counter += 1
            try:
                shutil.copy(audio_path, dest)
                audio_path = dest
            except Exception:
                pass

        # Preprocess audio
        clean_audio = preprocess_audio(audio_path)
        
        # Transcribe and segment (7-12 segments enforced)
        transcript, segments = transcribe(clean_audio)
        
        # Extract summaries for all segments
        segment_texts = [seg["text"] for seg in segments]
        summaries = summarize_segments(segment_texts)
        
        # Extract keywords (5 per segment)
        keywords_list = []
        for text in segment_texts:
            keywords = extract_keywords(text, num_keywords=5)
            keywords_list.append(keywords)
        
        # Compute sentiment scores
        sentiment_score, sentiments = avg_sentiment(segment_texts)
        
        # Create segment index with embeddings
        index = SegmentIndex()
        for i, seg in enumerate(segments):
            index.add_segment(
                segment_id=i + 1,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                keywords=keywords_list[i],
                summary=summaries[i] if i < len(summaries) else "",
                sentiment_score=float(sentiments[i]) if i < len(sentiments) else 0.0
            )
        
        # Compute embeddings
        index.compute_embeddings()
        
        # Create search engine
        search_engine = SegmentSearch(index)
        
        # Build topics for UI
        topics = []
        for i, seg in enumerate(segments):
            topics.append({
                "id": i + 1,
                "segment": seg["text"],
                "keywords": keywords_list[i],
                "summary": summaries[i] if i < len(summaries) else "",
                "topic": f"Topic {i+1}",
                "sentiment": float(sentiments[i]) if i < len(sentiments) else 0.0,
                "start": seg["start"],
                "end": seg["end"]
            })
        
        # Add topic info back to segments
        for i, seg in enumerate(index.segments):
            seg["topic"] = f"Topic {i+1}"
        
        # Save outputs if requested
        if save_output:
            try:
                output_dir_path = output_dir
                os.makedirs(output_dir_path, exist_ok=True)
                
                # Get episode ID from filename
                base_name = os.path.basename(audio_path)
                episode_id = os.path.splitext(base_name)[0]
                
                # Save segment data
                segment_file_path = os.path.join(output_dir_path, f"{episode_id}.json")
                index.save_to_json(segment_file_path)
                
                # Save embeddings separately
                embeddings_data = {
                    "episode_id": episode_id,
                    "embeddings": [seg["embedding"] for seg in index.segments if seg["embedding"]]
                }
                embeddings_file_path = os.path.join(output_dir_path, f"{episode_id}_embeddings.json")
                with open(embeddings_file_path, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, ensure_ascii=False)
                print(f" Embeddings saved: {embeddings_file_path}")
            except Exception as save_err:
                print(f"  Save warning: {str(save_err)}")
        
        return transcript, topics, sentiment_score, summaries, index, search_engine
    
    except Exception as e:
        print(f" Pipeline error: {str(e)}")
        raise