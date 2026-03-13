"""System testing for podcast analysis pipeline."""

import os
import json
from pathlib import Path
from datetime import datetime
from pipeline import run_pipeline


class SystemTester:
    """Test the podcast analysis system on multiple episodes."""
    
    def __init__(self, audio_dir, output_dir="test_results"):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def find_audio_files(self):
        """Find all audio files in the directory."""
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(self.audio_dir).glob(f"**/*{ext}"))
        
        return sorted(audio_files)
    
    def test_episode(self, audio_path):
        """Test pipeline on a single episode."""
        try:
            episode_name = Path(audio_path).stem
            print(f"\n{'='*60}")
            print(f"Testing: {episode_name}")
            print(f"{'='*60}")
            
            # Run pipeline
            transcript, topics, sentiment_score, summaries, index, search_engine = run_pipeline(str(audio_path))
            
            # Validate results
            validation = self._validate_results(
                transcript, topics, sentiment_score, summaries, index, search_engine
            )
            
            # Create test report
            report = {
                "episode": episode_name,
                "timestamp": datetime.now().isoformat(),
                "audio_path": str(audio_path),
                "validation": validation,
                "metrics": {
                    "total_segments": len(topics),
                    "avg_sentiment": sentiment_score,
                    "transcript_length": len(transcript),
                    "total_keywords": sum(len(t["keywords"]) for t in topics),
                    "avg_keywords_per_segment": sum(len(t["keywords"]) for t in topics) / len(topics) if topics else 0,
                    "total_duration": index.segments[-1]["end"] if index.segments else 0
                },
                "segment_details": self._extract_segment_details(index)
            }
            
            self.results.append(report)
            
            # Print summary
            self._print_report_summary(report)
            
            return report
            
        except Exception as e:
            error_report = {
                "episode": Path(audio_path).stem,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_report)
            print(f"❌ ERROR: {e}")
            return error_report
    
    def _validate_results(self, transcript, topics, sentiment_score, summaries, index, search_engine):
        """Validate pipeline results."""
        validations = {
            "transcript_not_empty": bool(transcript and len(transcript) > 0),
            "topics_in_range": 7 <= len(topics) <= 12,
            "topics_have_keywords": all(len(t.get("keywords", [])) >= 5 for t in topics),
            "topics_have_summaries": all(t.get("summary") for t in topics),
            "sentiment_in_range": -1.0 <= sentiment_score <= 1.0,
            "index_has_embeddings": all(seg.get("embedding") for seg in index.segments),
            "search_functional": len(search_engine.keyword_search("test", top_k=5)) >= 0
        }
        
        return validations
    
    def _extract_segment_details(self, index):
        """Extract detailed segment information."""
        details = []
        for seg in index.segments:
            details.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "keywords": seg["keywords"],
                "sentiment": seg["sentiment_score"],
                "text_preview": seg.get("segments", seg.get("text", ""))[:100] + "..." if len(seg.get("segments", seg.get("text", ""))) > 100 else seg.get("segments", seg.get("text", ""))
            })
        return details
    
    def _print_report_summary(self, report):
        """Print a summary of the test report."""
        if "error" in report:
            print(f"❌ Failed: {report['error']}")
            return
        
        metrics = report["metrics"]
        validation = report["validation"]
        
        print(f"\n✅ Result Validation:")
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}: {passed}")
        
        print(f"\n📊 Metrics:")
        print(f"  • Segments: {metrics['total_segments']}")
        print(f"  • Avg Sentiment: {metrics['avg_sentiment']:.3f}")
        print(f"  • Total Keywords: {metrics['total_keywords']}")
        print(f"  • Keywords/Segment: {metrics['avg_keywords_per_segment']:.1f}")
        print(f"  • Duration: {metrics['total_duration']:.1f}s")
    
    def run_tests(self, limit=None):
        """Run tests on all audio files."""
        audio_files = self.find_audio_files()
        
        if limit:
            audio_files = audio_files[:limit]
        
        print(f"\n🎙️ Found {len(audio_files)} audio file(s)")
        
        for audio_path in audio_files:
            self.test_episode(audio_path)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save test results to JSON."""
        output_file = os.path.join(self.output_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        self._print_summary()
    
    def _print_summary(self):
        """Print overall test summary."""
        print(f"\n{'='*60}")
        print("📋 TEST SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.results)
        successful = sum(1 for r in self.results if "error" not in r)
        failed = total - successful
        
        print(f"Total Tests: {total}")
        print(f"Successful: {successful} ")
        print(f"Failed: {failed} ❌")
        
        if successful > 0:
            avg_segments = sum(r.get("metrics", {}).get("total_segments", 0) 
                             for r in self.results if "error" not in r) / successful
            avg_sentiment = sum(r.get("metrics", {}).get("avg_sentiment", 0) 
                              for r in self.results if "error" not in r) / successful
            
            print(f"\nAverage Segments: {avg_segments:.1f}")
            print(f"Average Sentiment: {avg_sentiment:.3f}")


if __name__ == "__main__":
    import sys
    
    # Get audio directory from command line or use default
    audio_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw_audio"
    
    # Limit tests (for quick testing)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Run tests
    tester = SystemTester(audio_dir)
    tester.run_tests(limit=limit)
