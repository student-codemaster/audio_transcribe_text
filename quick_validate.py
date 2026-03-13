"""Quick validation - checks files without heavy imports."""

import os
from pathlib import Path


def validate_files():
    """Validate all required files exist."""
    print("\n" + "="*70)
    print("🔍 AI AUDIO TRANSCRIBER v2.0 - QUICK VALIDATION")
    print("="*70)
    
    base_path = Path(__file__).parent
    
    # Files to check
    checks = {
        '🎙️ Core Files': [
            'pipeline.py',
            'segment_indexing.py',
            'visualization.py',
            'sentiment.py',
            'keywords.py',
        ],
        '✨ NEW - Enhanced App & Testing': [
            'streamlit_app_v2.py',
            'multi_episode_test.py',
        ],
        '📚 NEW - Documentation': [
            'VISUALIZATION_GUIDE.md',
            'QUICK_REFERENCE.md',
            'UPDATE_SUMMARY.md',
            'validate_project.py',
        ],
        '⚙️ Configuration': [
            'requirements.txt',
        ]
    }
    
    total_passed = 0
    total_checks = 0
    
    for category, files in checks.items():
        print(f"\n{category}")
        print("-" * 70)
        for filename in files:
            filepath = base_path / filename
            total_checks += 1
            
            if filepath.exists():
                size = filepath.stat().st_size
                size_kb = size / 1024
                print(f"  ✅ {filename:<35} ({size_kb:>6.1f} KB)")
                total_passed += 1
            else:
                print(f"  ❌ {filename:<35} NOT FOUND")
    
    # Check visualization functions
    print(f"\n🎨 Visualization Functions")
    print("-" * 70)
    
    viz_path = base_path / 'visualization.py'
    if viz_path.exists():
        with open(viz_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        functions = [
            'create_segment_timeline',
            'create_sentiment_trend',
            'create_keyword_cloud',
            'create_keyword_bar_chart',
            'create_segment_distribution',
            'create_sentiment_heatmap',
            'create_keywords_per_segment',
        ]
        
        for func in functions:
            if f'def {func}' in content:
                print(f"  ✅ {func}")
                total_passed += 1
            else:
                print(f"  ❌ {func}")
            total_checks += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"✅ PASSED: {total_passed}/{total_checks}")
    print(f"📊 SUCCESS RATE: {(total_passed/total_checks*100):.1f}%")
    print("="*70)
    
    if total_passed == total_checks:
        print("\n🎉 ALL SYSTEMS GO! Project is ready for use.")
        print("\n📋 Next Steps:")
        print("   1. Run: streamlit run streamlit_app_v2.py")
        print("   2. Upload a podcast for analysis")
        print("   3. Explore all visualizations")
        print("   4. Try: python multi_episode_test.py data/")
        return True
    else:
        print(f"\n⚠️ {total_checks - total_passed} issues found - please review above")
        return False


if __name__ == '__main__':
    import sys
    success = validate_files()
    sys.exit(0 if success else 1)
