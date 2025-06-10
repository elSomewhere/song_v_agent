"""
Test script to verify the storyboard generator setup.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import openai
        print("✓ OpenAI")
    except ImportError as e:
        print(f"✗ OpenAI: {e}")
        return False
    
    try:
        import lancedb
        print("✓ LanceDB")
    except ImportError as e:
        print(f"✗ LanceDB: {e}")
        return False
    
    try:
        from langgraph.graph import StateGraph
        print("✓ LangGraph")
    except ImportError as e:
        print(f"✗ LangGraph: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers")
    except ImportError as e:
        print(f"✗ Sentence Transformers: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow")
    except ImportError as e:
        print(f"✗ PIL/Pillow: {e}")
        return False
    
    try:
        import rich
        print("✓ Rich")
    except ImportError as e:
        print(f"✗ Rich: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\nTesting environment...")
    
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OpenAI API key found (length: {len(api_key)})")
        return True
    else:
        print("✗ OpenAI API key not found")
        print("  Please create a .env file with OPENAI_API_KEY=your-key")
        return False

def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")
    
    success = True
    
    # Check script
    script_path = Path("data/script.md")
    if script_path.exists():
        print(f"✓ Script found: {script_path}")
    else:
        print(f"✗ Script not found: {script_path}")
        success = False
    
    # Check style primer
    style_path = Path("data/style_primer.md")
    if style_path.exists():
        print(f"✓ Style primer found: {style_path}")
    else:
        print(f"✗ Style primer not found: {style_path}")
        success = False
    
    # Check refs directory
    refs_path = Path("data/refs")
    if refs_path.exists() and refs_path.is_dir():
        subdirs = list(refs_path.iterdir())
        print(f"✓ Refs directory found with {len(subdirs)} entities")
        for subdir in subdirs:
            if subdir.is_dir():
                images = list(subdir.glob("*"))
                print(f"  - {subdir.name}: {len(images)} files")
    else:
        print(f"✗ Refs directory not found: {refs_path}")
        success = False
    
    return success

def test_agents():
    """Test that agent modules can be imported."""
    print("\nTesting agent modules...")
    
    try:
        from agents import shared, director, artist, critic
        print("✓ All agent modules can be imported")
        
        # Test shared utilities
        style = shared.load_style_primer()
        print(f"✓ Style primer loaded ({len(style)} chars)")
        
        return True
    except Exception as e:
        print(f"✗ Error loading agents: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Storyboard Generator Setup Test ===\n")
    
    all_good = True
    
    if not test_imports():
        all_good = False
        print("\n⚠️  Some imports failed. Run: pip install -r requirements.txt")
    
    if not test_environment():
        all_good = False
        print("\n⚠️  Environment setup incomplete")
    
    if not test_data_files():
        all_good = False
        print("\n⚠️  Data files missing or incomplete")
    
    if not test_agents():
        all_good = False
        print("\n⚠️  Agent modules have issues")
    
    print("\n" + "="*40)
    if all_good:
        print("✅ All tests passed! Ready to run:")
        print("   1. python ingest/ingest_text.py")
        print("   2. python ingest/ingest_refs.py")
        print("   3. python run.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 