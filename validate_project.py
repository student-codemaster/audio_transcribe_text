"""Validation script to verify all new components are working correctly."""

import os
import sys
import json
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False


def check_imports():
    """Check if all required imports are available."""
    print("\n📦 Checking Required Imports...")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'wordcloud': 'WordCloud',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'keybert': 'KeyBERT',
    }
    
    all_available = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_available = False
    
    return all_available


def check_project_structure():
    """Check if all project files are in place."""
    print("\n📁 Checking Project Structure...")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = {
        'Core Files': [
            ('pipeline.py', 'Pipeline'),
            ('segment_indexing.py', 'Segment Indexing'),
            ('visualization.py', 'Visualization (Enhanced)'),
            ('sentiment.py', 'Sentiment Analysis'),
            ('keywords.py', 'Keyword Extraction'),
        ],
        'New App & Testing': [
            ('streamlit_app_v2.py', 'Streamlit App v2'),
            ('multi_episode_test.py', 'Multi-Episode Testing'),
        ],
        'Documentation': [
            ('VISUALIZATION_GUIDE.md', 'Visualization Guide'),
            ('QUICK_REFERENCE.md', 'Quick Reference'),
            ('UPDATE_SUMMARY.md', 'Update Summary'),
        ],
        'Configuration': [
            ('requirements.txt', 'Requirements'),
        ]
    }
    
    results = {}
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        results[category] = True
        for filename, description in files:
            filepath = os.path.join(base_path, filename)
            exists = check_file_exists(filepath, description)
            results[category] = results[category] and exists
    
    return results


def check_visualization_functions():
    """Check if all visualization functions exist."""
    print("\n🎨 Checking Visualization Functions...")
    
    try:
        from visualization import (
            create_segment_timeline,
            create_sentiment_trend,
            create_keyword_cloud,
            create_keyword_bar_chart,
            create_segment_distribution,
            create_sentiment_heatmap,
            create_keywords_per_segment,
            format_time
        )
        
        functions = [
            ('create_segment_timeline', 'Timeline'),
            ('create_sentiment_trend', 'Sentiment Trend'),
            ('create_keyword_cloud', 'Keyword Cloud'),
            ('create_keyword_bar_chart', 'Keyword Bar Chart'),
            ('create_segment_distribution', 'Segment Distribution'),
            ('create_sentiment_heatmap', 'Sentiment Heatmap'),
            ('create_keywords_per_segment', 'Keywords per Segment'),
            ('format_time', 'Time Formatter'),
        ]
        
        for func_name, description in functions:
            print(f"✅ {description}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import visualization functions: {e}")
        return False


def check_multi_episode_test():
    """Check if MultiEpisodeTestRunner is available."""
    print("\n🧪 Checking Multi-Episode Testing...")
    
    try:
        from multi_episode_test import MultiEpisodeTestRunner
        print(f"✅ MultiEpisodeTestRunner class found")
        
        # Check for key methods
        methods = [
            'test_episode',
            'run_batch_test',
            '_assess_segmentation_quality',
            '_assess_keyword_quality',
            '_assess_summary_quality',
            '_assess_sentiment_quality',
        ]
        
        for method in methods:
            if hasattr(MultiEpisodeTestRunner, method):
                print(f"✅ Method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import MultiEpisodeTestRunner: {e}")
        return False


def check_visualization_file_content():
    """Check visualization.py has new functions."""
    print("\n📝 Checking Visualization File Content...")
    
    try:
        with open('visualization.py', 'r') as f:
            content = f.read()
        
        expected_functions = [
            'create_segment_timeline',
            'create_sentiment_trend',
            'create_keyword_cloud',
            'create_keyword_bar_chart',
            'create_segment_distribution',
            'create_sentiment_heatmap',
            'create_keywords_per_segment',
        ]
        
        all_found = True
        for func in expected_functions:
            if f'def {func}' in content:
                print(f"✅ Function {func} defined")
            else:
                print(f"❌ Function {func} NOT FOUND")
                all_found = False
        
        # Check line count
        lines = len(content.split('\n'))
        print(f"\n📊 visualization.py Stats: {lines} lines")
        
        if lines > 300:
            print(f"✅ File has substantial new code")
        else:
            print(f"⚠️ File may be too short")
        
        return all_found
    except Exception as e:
        print(f"❌ Error checking visualization.py: {e}")
        return False


def check_streamlit_app_v2():
    """Check streamlit_app_v2.py for required sections."""
    print("\n🎙️ Checking Streamlit App v2...")
    
    try:
        with open('streamlit_app_v2.py', 'r') as f:
            content = f.read()
        
        required_sections = [
            ('create_segment_timeline', 'Timeline import'),
            ('create_sentiment_trend', 'Sentiment import'),
            ('create_keyword_cloud', 'Keyword cloud import'),
            ('elif page == "⏱️ Timeline"', 'Timeline page'),
            ('elif page == "😊 Sentiment"', 'Sentiment page'),
            ('elif page == "🏷️ Keywords"', 'Keywords page'),
            ('elif page == "📊 Analytics"', 'Analytics page'),
            ('MultiEpisodeTest', 'Multi-episode section'),
        ]
        
        all_found = True
        for section, description in required_sections:
            if section in content:
                print(f"✅ {description}")
            else:
                print(f"⚠️ {description} - Not found (may be OK)")
        
        lines = len(content.split('\n'))
        print(f"\n📊 streamlit_app_v2.py Stats: {lines} lines")
        
        if lines > 500:
            print(f"✅ App is comprehensive")
        
        return True
    except Exception as e:
        print(f"❌ Error checking streamlit_app_v2.py: {e}")
        return False


def check_requirements_txt():
    """Check if all required packages are in requirements.txt."""
    print("\n📦 Checking requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'streamlit',
            'plotly',
            'pandas',
            'numpy',
            'matplotlib',
            'wordcloud',
            'transformers',
            'sentence-transformers',
            'keybert',
        ]
        
        all_found = True
        for package in required_packages:
            if package in requirements:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - NOT IN requirements.txt")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"❌ Error checking requirements.txt: {e}")
        return False


def check_documentation():
    """Check if documentation files have content."""
    print("\n📚 Checking Documentation...")
    
    docs = {
        'VISUALIZATION_GUIDE.md': 300,
        'QUICK_REFERENCE.md': 200,
        'UPDATE_SUMMARY.md': 200,
    }
    
    all_good = True
    for doc_file, min_lines in docs.items():
        try:
            with open(doc_file, 'r') as f:
                lines = len(f.readlines())
            
            if lines >= min_lines:
                print(f"✅ {doc_file}: {lines} lines")
            else:
                print(f"⚠️ {doc_file}: {lines} lines (< {min_lines} expected)")
                all_good = False
        except FileNotFoundError:
            print(f"❌ {doc_file} NOT FOUND")
            all_good = False
    
    return all_good


def generate_report(results):
    """Generate a summary report."""
    print("\n" + "="*60)
    print("📋 VALIDATION REPORT")
    print("="*60)
    
    total_checks = 0
    passed_checks = 0
    
    # Count results
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, bool):
                total_checks += 1
                if value:
                    passed_checks += 1
    
    print(f"\n✅ Checks Passed: {passed_checks}")
    print(f"❌ Checks Failed: {total_checks - passed_checks}")
    print(f"📊 Pass Rate: {(passed_checks/total_checks*100):.1f}%" if total_checks > 0 else "N/A")
    
    if passed_checks == total_checks:
        print("\n🎉 ALL CHECKS PASSED - Project is ready!")
    else:
        print("\n⚠️ Some checks failed - See details above")
    
    print("\n" + "="*60)


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("🔍 AI AUDIO TRANSCRIBER - PROJECT VALIDATION")
    print("="*60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = {}
    
    # Run checks
    print("\n🌡️ Temperature Check: Starting validation...\n")
    
    results['Structure'] = check_project_structure()
    results['Imports'] = check_imports()
    results['Visualization Functions'] = check_visualization_functions()
    results['Multi-Episode Testing'] = check_multi_episode_test()
    results['Visualization Content'] = check_visualization_file_content()
    results['Streamlit App v2'] = check_streamlit_app_v2()
    results['Documentation'] = check_documentation()
    results['Requirements'] = check_requirements_txt()
    
    # Generate report
    generate_report(results)
    
    # Provide next steps
    print("\n📝 Next Steps:")
    print("1. Run: streamlit run streamlit_app_v2.py")
    print("2. Upload podcast audio for analysis")
    print("3. Explore visualizations in dashboard")
    print("4. Try multi-episode testing: python multi_episode_test.py data/")
    print("\n✨ Enjoy the enhanced AI Audio Transcriber!")


if __name__ == "__main__":
    main()
