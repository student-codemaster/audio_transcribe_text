"""Multi-episode testing and quality assessment for podcast analysis pipeline."""

import json
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from pipeline import run_pipeline


class MultiEpisodeTestRunner:
    """Test the pipeline on multiple episodes and generate quality reports."""
    
    def __init__(self, output_dir="test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.test_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def test_episode(self, audio_path):
        """Test pipeline on a single episode.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with test results
        """
        episode_name = Path(audio_path).stem
        print(f"\n{'='*60}")
        print(f"Testing Episode: {episode_name}")
        print(f"{'='*60}")
        
        try:
            # Run pipeline
            transcript, topics, sentiment_score, summaries, index, search_engine = run_pipeline(
                audio_path, save_output=True
            )
            
            # Assess segmentation quality
            segmentation_quality = self._assess_segmentation_quality(index.segments)
            
            # Assess keyword relevance
            keyword_quality = self._assess_keyword_quality(index.segments)
            
            # Assess summary accuracy
            summary_quality = self._assess_summary_quality(index.segments)
            
            # Assess sentiment consistency
            sentiment_quality = self._assess_sentiment_quality(index.segments)
            
            result = {
                'episode_name': episode_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS',
                'metrics': {
                    'total_segments': len(index.segments),
                    'total_duration': float(index.segments[-1]['end']) if index.segments else 0,
                    'avg_segment_duration': float(np.mean([s['duration'] for s in index.segments])) if index.segments else 0,
                    'sentiment_score': float(sentiment_score),
                    'total_keywords': sum(len(s['keywords']) for s in index.segments),
                    'unique_keywords': len(set(kw for s in index.segments for kw in s['keywords']))
                },
                'quality_assessment': {
                    'segmentation': segmentation_quality,
                    'keywords': keyword_quality,
                    'summaries': summary_quality,
                    'sentiment': sentiment_quality
                },
                'observations': self._generate_observations(index.segments, sentiment_score)
            }
            
            print(f"\n✅ Episode processed successfully")
            self._print_results(result)
            
            return result
            
        except Exception as e:
            result = {
                'episode_name': episode_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'FAILED',
                'error': str(e),
                'metrics': {},
                'quality_assessment': {}
            }
            print(f"\n❌ Error processing episode: {str(e)}")
            return result
    
    def _assess_segmentation_quality(self, segments):
        """Assess quality of segmentation.
        
        Returns:
            Dictionary with quality metrics
        """
        if not segments:
            return {'score': 0, 'details': 'No segments'}
        
        durations = [s['duration'] for s in segments]
        
        # Calculate metrics
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        cv = std_duration / avg_duration if avg_duration > 0 else 0  # Coefficient of variation
        
        # Score based on consistency (lower CV is better, but not too low)
        if cv < 0.3:
            consistency_score = 1.0
        elif cv < 0.5:
            consistency_score = 0.8
        elif cv < 0.7:
            consistency_score = 0.6
        else:
            consistency_score = 0.4
        
        # Check for very short or very long segments
        short_segments = sum(1 for d in durations if d < 10)
        long_segments = sum(1 for d in durations if d > 300)
        outlier_penalty = (short_segments + long_segments) * 0.05
        
        quality_score = max(consistency_score - outlier_penalty, 0)
        
        return {
            'score': round(quality_score, 3),
            'total_segments': len(segments),
            'avg_duration': round(avg_duration, 1),
            'duration_std_dev': round(std_duration, 1),
            'consistency_cv': round(cv, 3),
            'short_segments': short_segments,
            'long_segments': long_segments
        }
    
    def _assess_keyword_quality(self, segments):
        """Assess quality of keyword extraction.
        
        Returns:
            Dictionary with quality metrics
        """
        if not segments:
            return {'score': 0, 'details': 'No segments'}
        
        all_keywords = []
        keywords_per_segment = []
        
        for seg in segments:
            keywords = seg.get('keywords', [])
            all_keywords.extend(keywords)
            keywords_per_segment.append(len(keywords))
        
        # Calculate metrics
        unique_ratio = len(set(all_keywords)) / len(all_keywords) if all_keywords else 0
        avg_keywords = np.mean(keywords_per_segment)
        
        # Assess keyword diversity
        if unique_ratio > 0.7:
            diversity_score = 1.0
        elif unique_ratio > 0.5:
            diversity_score = 0.8
        elif unique_ratio > 0.3:
            diversity_score = 0.6
        else:
            diversity_score = 0.4
        
        # Assess keyword coverage
        no_keywords_count = sum(1 for k in keywords_per_segment if k == 0)
        coverage_score = 1.0 - (no_keywords_count / len(segments))
        
        quality_score = (diversity_score * 0.6 + coverage_score * 0.4)
        
        return {
            'score': round(quality_score, 3),
            'total_keywords': len(all_keywords),
            'unique_keywords': len(set(all_keywords)),
            'diversity_ratio': round(unique_ratio, 3),
            'avg_keywords_per_segment': round(avg_keywords, 1),
            'segments_without_keywords': no_keywords_count
        }
    
    def _assess_summary_quality(self, segments):
        """Assess quality of summaries.
        
        Returns:
            Dictionary with quality metrics
        """
        if not segments:
            return {'score': 0, 'details': 'No segments'}
        
        summary_lengths = []
        text_lengths = []
        
        for seg in segments:
            summary = seg.get('summary', '')
            text = seg.get('segments', seg.get('text', ''))
            
            summary_words = len(summary.split())
            text_words = len(text.split())
            
            summary_lengths.append(summary_words)
            text_lengths.append(text_words if text_words > 0 else 1)
        
        # Calculate compression ratios
        compression_ratios = []
        for i in range(len(segments)):
            if text_lengths[i] > 0:
                ratio = summary_lengths[i] / text_lengths[i]
                compression_ratios.append(ratio)
        
        avg_ratio = np.mean(compression_ratios) if compression_ratios else 0
        
        # Good compression ratio is between 0.1 and 0.5
        if 0.1 <= avg_ratio <= 0.5:
            compression_score = 1.0
        elif 0.05 <= avg_ratio <= 0.7:
            compression_score = 0.8
        elif avg_ratio < 0.1:
            compression_score = 0.6  # Summaries too short
        else:
            compression_score = 0.5  # Summaries too long
        
        # Check coverage
        empty_summaries = sum(1 for s in summary_lengths if s == 0)
        coverage_score = 1.0 - (empty_summaries / len(segments))
        
        quality_score = (compression_score * 0.7 + coverage_score * 0.3)
        
        return {
            'score': round(quality_score, 3),
            'avg_summary_length': round(np.mean(summary_lengths), 1),
            'avg_compression_ratio': round(avg_ratio, 3),
            'empty_summaries': empty_summaries,
            'coverage_score': round(coverage_score, 3)
        }
    
    def _assess_sentiment_quality(self, segments):
        """Assess quality of sentiment analysis.
        
        Returns:
            Dictionary with quality metrics
        """
        if not segments:
            return {'score': 0, 'details': 'No segments'}
        
        sentiments = [s.get('sentiment_score', 0) for s in segments]
        
        # Check if sentiments are reasonable values
        valid_sentiments = all(-1.0 <= s <= 1.0 for s in sentiments)
        
        if not valid_sentiments:
            return {'score': 0.3, 'details': 'Invalid sentiment values detected'}
        
        # Assess sentiment distribution
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)
        
        positive = sum(1 for s in sentiments if s > 0.5)
        neutral = sum(1 for s in sentiments if -0.5 <= s <= 0.5)
        negative = sum(1 for s in sentiments if s < -0.5)
        
        total = len(sentiments)
        
        # Good distribution should have reasonable balance
        distribution_score = 0.7  # Default
        
        if neutral > (total * 0.5):  # More than 50% neutral is expected
            distribution_score = 0.9
        
        # Variance score (some variation is good)
        if std_sentiment > 0.1:
            variance_score = 0.85
        elif std_sentiment > 0.01:
            variance_score = 0.7
        else:
            variance_score = 0.5  # Very little variation might indicate issues
        
        quality_score = (distribution_score * 0.6 + variance_score * 0.4)
        
        return {
            'score': round(quality_score, 3),
            'mean_sentiment': round(mean_sentiment, 3),
            'std_sentiment': round(std_sentiment, 3),
            'positive_segments': positive,
            'neutral_segments': neutral,
            'negative_segments': negative,
            'distribution': {
                'positive_pct': round(positive/total*100, 1),
                'neutral_pct': round(neutral/total*100, 1),
                'negative_pct': round(negative/total*100, 1)
            }
        }
    
    def _generate_observations(self, segments, sentiment_score):
        """Generate human-readable observations about the analysis.
        
        Returns:
            List of observation strings
        """
        observations = []
        
        # Segment observations
        if len(segments) < 5:
            observations.append("⚠️ Few segments detected - content might be short or homogeneous")
        elif len(segments) > 20:
            observations.append("✓ Good segmentation granularity")
        
        # Keyword observations
        all_keywords = [kw for s in segments for kw in s['keywords']]
        unique_kw = len(set(all_keywords))
        
        if unique_kw < 5:
            observations.append("⚠️ Limited keyword diversity")
        elif unique_kw > 30:
            observations.append("✓ Rich keyword diversity detected")
        
        # Sentiment observations
        if sentiment_score > 0.6:
            observations.append("😊 Overall positive sentiment detected")
        elif sentiment_score < -0.4:
            observations.append("😞 Overall negative sentiment detected")
        else:
            observations.append("😐 Balanced/neutral sentiment throughout")
        
        # Duration observations
        durations = [s['duration'] for s in segments]
        max_duration = max(durations) if durations else 0
        
        if max_duration > 600:
            observations.append("⚠️ Some very long segments detected - consider refining segmentation")
        
        # Summary observations
        summaries = [s.get('summary', '') for s in segments]
        empty_summaries = sum(1 for s in summaries if not s or len(s.strip()) < 10)
        
        if empty_summaries > len(segments) * 0.3:
            observations.append("⚠️ Many empty or very short summaries")
        else:
            observations.append("✓ Good summary coverage")
        
        return observations
    
    def _print_results(self, result):
        """Print formatted test results."""
        print(f"\n📊 Quality Assessment:")
        for category, metrics in result.get('quality_assessment', {}).items():
            score = metrics.get('score', 'N/A')
            print(f"  • {category.capitalize()}: {score}")
        
        print(f"\n📝 Observations:")
        for obs in result.get('observations', []):
            print(f"  {obs}")
    
    def run_batch_test(self, audio_folder):
        """Run tests on all audio files in a folder.
        
        Args:
            audio_folder: Path to folder containing audio files
        
        Returns:
            List of test results
        """
        audio_files = glob.glob(os.path.join(audio_folder, "*.wav")) + \
                     glob.glob(os.path.join(audio_folder, "*.mp3"))
        
        print(f"\n{'='*60}")
        print(f"🎙️ Multi-Episode Test Runner")
        print(f"Found {len(audio_files)} episode(s)")
        print(f"{'='*60}")
        
        for audio_path in audio_files:
            result = self.test_episode(audio_path)
            self.test_results.append(result)
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.test_results
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        successful_tests = [r for r in self.test_results if r['status'] == 'SUCCESS']
        
        if not successful_tests:
            print("\n❌ No successful tests to report")
            return
        
        print(f"\n{'='*60}")
        print(f"📋 Summary Report")
        print(f"{'='*60}")
        
        # Aggregate metrics
        total_segments = sum(r['metrics'].get('total_segments', 0) for r in successful_tests)
        avg_segment_duration = np.mean([r['metrics'].get('avg_segment_duration', 0) 
                                       for r in successful_tests if r['metrics'].get('avg_segment_duration')])
        avg_sentiment = np.mean([r['metrics'].get('sentiment_score', 0) 
                               for r in successful_tests if r['metrics'].get('sentiment_score')])
        
        print(f"\n📊 Aggregate Metrics Across {len(successful_tests)} Episodes:")
        print(f"  • Total Segments: {total_segments}")
        print(f"  • Avg Segment Duration: {avg_segment_duration:.1f}s")
        print(f"  • Avg Sentiment Score: {avg_sentiment:.3f}")
        
        # Quality scores
        print(f"\n⭐ Average Quality Scores:")
        
        qualities = {}
        for quality_type in ['segmentation', 'keywords', 'summaries', 'sentiment']:
            scores = [r['quality_assessment'].get(quality_type, {}).get('score', 0) 
                     for r in successful_tests]
            avg_score = np.mean(scores) if scores else 0
            qualities[quality_type] = avg_score
            quality_bar = '█' * int(avg_score * 10) + '░' * (10 - int(avg_score * 10))
            print(f"  • {quality_type.capitalize()}: {avg_score:.2f} {quality_bar}")
        
        # Save report
        self._save_report(successful_tests, qualities)
    
    def _save_report(self, results, quality_summary):
        """Save detailed test report to JSON."""
        report = {
            'timestamp': self.timestamp,
            'total_tests': len(results),
            'quality_summary': quality_summary,
            'detailed_results': results
        }
        
        report_path = os.path.join(self.output_dir, f"test_report_{self.timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Report saved: {report_path}")
        
        # Also save CSV summary
        df_summary = pd.DataFrame([
            {
                'Episode': r['episode_name'],
                'Status': r['status'],
                'Segments': r['metrics'].get('total_segments', 0),
                'Duration': r['metrics'].get('total_duration', 0),
                'Avg_Segment_Duration': r['metrics'].get('avg_segment_duration', 0),
                'Segmentation_Score': r['quality_assessment'].get('segmentation', {}).get('score', 0),
                'Keywords_Score': r['quality_assessment'].get('keywords', {}).get('score', 0),
                'Summaries_Score': r['quality_assessment'].get('summaries', {}).get('score', 0),
                'Sentiment_Score': r['quality_assessment'].get('sentiment', {}).get('score', 0)
            }
            for r in results
        ])
        
        csv_path = os.path.join(self.output_dir, f"summary_{self.timestamp}.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"💾 Summary CSV saved: {csv_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get audio folder from command line or use default
    if len(sys.argv) > 1:
        audio_folder = sys.argv[1]
    else:
        audio_folder = "data"  # Default folder
    
    # Run tests
    runner = MultiEpisodeTestRunner(output_dir="test_results")
    
    if os.path.isfile(audio_folder):
        # Single file test
        runner.test_episode(audio_folder)
    else:
        # Batch test
        runner.run_batch_test(audio_folder)
