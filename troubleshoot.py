"""Network timeout troubleshooting and setup guide."""

import os
from pathlib import Path

def setup_model_cache():
    """Setup hugging face model cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['HF_HUB_CACHE'] = str(cache_dir)
    
    print(f"✅ Cache directory set: {cache_dir}")
    return str(cache_dir)


def test_imports():
    """Test if all required imports work."""
    print("\n🔍 Testing imports...")
    
    issues = []
    
    try:
        import streamlit
        print("streamlit")
    except ImportError as e:
        issues.append(f"streamlit: {e}")
    
    try:
        import plotly
        print(" plotly")
    except ImportError as e:
        issues.append(f"plotly: {e}")
    
    try:
        import pandas
        print(" pandas")
    except ImportError as e:
        issues.append(f"pandas: {e}")
    
    try:
        import wordcloud
        print(" wordcloud")
    except ImportError as e:
        issues.append(f"wordcloud: {e}")
    
    try:
        import sentence_transformers
        print(" sentence_transformers")
    except ImportError as e:
        issues.append(f"sentence_transformers: {e}")
    
    try:
        # This won't load the model, just the module
        from embedding_model import get_model
        print(" embedding_model")
    except Exception as e:
        issues.append(f"embedding_model: {e}")
    
    try:
        from pipeline import run_pipeline
        print("pipeline")
    except Exception as e:
        issues.append(f"pipeline: {e}")
    
    if issues:
        print(f"\n {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n All imports successful!")
        return True


def test_model_loading():
    """Test if the embedding model can be loaded."""
    print("\n Testing model loading...")
    
    try:
        setup_model_cache()
        from embedding_model import get_model
        
        print(" Loading embedding model (this may take a minute on first run)...")
        model = get_model()
        
        if model is not None:
            print(" Model loaded successfully!")
            return True
        else:
            print(" Model returned None (network timeout likely)")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def troubleshoot():
    """Run troubleshooting commands."""
    print("\n" + "="*60)
    print("🔧 AI AUDIO TRANSCRIBER - TROUBLESHOOTING")
    print("="*60)
    
    # Step 1: Setup
    print("\n📦 Step 1: Setting up model cache...")
    cache_dir = setup_model_cache()
    
    # Step 2: Test imports
    print("\n📦 Step 2: Testing imports...")
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️  Fix missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    # Step 3: Test model loading
    print("\n📦 Step 3: Testing model loading...")
    model_ok = test_model_loading()
    
    if not model_ok:
        print("\n💡 If model loading fails:")
        print("   1. Check your internet connection")
        print("   2. The model will auto-retry when you run the app")
        print("   3. You can manually pre-download with:")
        print(f"      python -c \"from embedding_model import get_model; get_model()\"")
    
    # Results
    print("\n" + "="*60)
    if imports_ok and model_ok:
        print("✅ Everything looks good!")
        print("\n🚀 Start the app with:")
        print("   streamlit run streamlit_app_v2.py")
    elif imports_ok:
        print("⚠️  Imports OK, but model loading needs network")
        print("   App will work with retries")
    else:
        print("❌ Issues found - see above for fixes")
    
    print("="*60)


if __name__ == "__main__":
    troubleshoot()
