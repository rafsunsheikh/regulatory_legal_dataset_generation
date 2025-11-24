"""Test script to verify Ollama setup and model availability."""

import sys
sys.path.insert(0, '.')

from src.generator import test_ollama_connection

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ollama Setup")
    print("=" * 60)
    
    if test_ollama_connection():
        print("\n✓ All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("1. Place your PDF files in: data/raw/")
        print("2. Run: python main.py")
    else:
        print("\n✗ Setup incomplete. Please follow these steps:")
        print("\n1. Start Ollama:")
        print("   ollama serve")
        print("\n2. In another terminal, pull the model:")
        print("   ollama pull llama3.1:8b")
        print("\n3. Run this test again:")
        print("   python test_setup.py")
    
    print("=" * 60)
