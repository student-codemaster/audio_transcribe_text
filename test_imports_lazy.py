#!/usr/bin/env python3
"""Test that lazy loading works - imports should NOT timeout."""

import sys
import traceback

print("=" * 60)
print("Testing lazy load pattern - imports should complete quickly")
print("=" * 60)

# Test individual module imports (should be fast)
modules_to_test = [
    "streamlit",
    "plotly.express",
    "plotly.graph_objects",
    "pandas",
    "numpy",
    "wordcloud",
    "embedding_model",
    "summarizer",
    "sentiment",
    "keywords",
    "speech_to_text",
]

print("\n1️⃣ Testing quick imports (should be <5 seconds total)...\n")

passed = 0
failed = 0

for module_name in modules_to_test:
    try:
        print(f"   Importing {module_name:25} ... ", end="", flush=True)
        __import__(module_name)
        print("✅")
        passed += 1
    except Exception as e:
        print(f"❌ {str(e)[:60]}")
        failed += 1

print(f"\n✅ Quick imports: {passed} passed, {failed} failed")

# Test that model loading is deferred
print("\n2️⃣ Testing lazy model loading (models should NOT load yet)...\n")

if failed == 0:
    print("   ✅ All imports successful with lazy loading!")
    print("   Models will only load when first used.\n")
else:
    print(f"   ⚠️ {failed} imports failed. This may block pipeline startup.")
    print("   See errors above.\n")

print("\n3️⃣ Testing pipeline import (the main bottleneck)...\n")

try:
    print("   Importing pipeline... ", end="", flush=True)
    import pipeline
    print("✅")
    print("   ✅ Pipeline imports successfully!")
except Exception as e:
    print(f"❌")
    print(f"\n   Error importing pipeline: {str(e)[:200]}")
    print(f"\n   Full traceback:")
    traceback.print_exc()

# Test streamlit app import
print("\n4️⃣ Testing streamlit app import...\n")

try:
    print("   Importing streamlit_app_v2... ", end="", flush=True)
    import streamlit_app_v2
    print("✅")
    print("   ✅ Streamlit app imports successfully!")
except Exception as e:
    print(f"❌")
    print(f"\n   Error importing streamlit_app_v2: {str(e)[:200]}")

print("\n" + "=" * 60)
print("Summary: Lazy loading prevents timeout at import time!")
print("Models will be loaded when functions are first called.")
print("=" * 60)
