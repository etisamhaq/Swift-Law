import streamlit as st
import json
import os
from unittest.mock import patch, MagicMock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from app
from app import validate_response, sanitize_input


def test_validate_response():
    """Test the response validation function."""
    print("\n=== Testing Response Validation ===")
    
    test_cases = [
        # Response with external knowledge indicators
        ("Based on general knowledge, the law states...", [], True),
        ("Typically, in Pakistan law...", [], True),
        
        # Short/generic responses
        ("Yes", [], True),
        ("No information found", [], True),
        
        # Valid responses
        ("According to Article 25 of the Constitution of Pakistan, all citizens are equal before law and are entitled to equal protection of law.", [], False),
        ("The Pakistan Penal Code Section 379 defines theft as whoever intending to take dishonestly any moveable property out of the possession of any person without that person's consent.", [], False),
    ]
    
    passed = 0
    for response, source_docs, should_be_modified in test_cases:
        result = validate_response(response, source_docs)
        was_modified = result != response
        status = "✓" if was_modified == should_be_modified else "✗"
        print(f"{status} Response modified: {was_modified} (Expected: {should_be_modified})")
        if was_modified == should_be_modified:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

def test_sanitize_input():
    """Test input sanitization function."""
    print("\n=== Testing Input Sanitization ===")
    
    test_cases = [
        # Normal input
        ("What is the law?", "What is the law?"),
        
        # Input with multiple spaces
        ("What   is    the   law?", "What is the law?"),
        
        # Input with special characters
        ("What <script>alert('test')</script> is the law?", "What scriptalert('test')/script is the law?"),
        
        # Long input (should be truncated)
        ("a" * 600, "a" * 500),
        
        # Input with newlines and tabs
        ("What\nis\tthe\nlaw?", "What is the law?"),
    ]
    
    passed = 0
    for input_text, expected in test_cases:
        result = sanitize_input(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: '{input_text[:50]}...' -> Expected: '{expected[:50]}...', Got: '{result[:50]}...'")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

def test_preprocessed_text_exists():
    """Test if preprocessed text file exists and is valid."""
    print("\n=== Testing Preprocessed Text File ===")
    
    if os.path.exists('preprocessed_text.json'):
        try:
            with open('preprocessed_text.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'text' in data and len(data['text']) > 0:
                    print(f"✓ Preprocessed text file exists with {len(data['text'])} characters")
                    return True
                else:
                    print("✗ Preprocessed text file exists but is empty or invalid")
                    return False
        except Exception as e:
            print(f"✗ Error reading preprocessed text file: {e}")
            return False
    else:
        print("✗ Preprocessed text file not found")
        return False

def test_api_key_configuration():
    """Test if API keys are properly configured."""
    print("\n=== Testing API Key Configuration ===")
    
    all_keys_found = True
    
    # Check Google API key
    if "GOOGLE_API_KEY" in os.environ:
        print("✓ GOOGLE_API_KEY found in environment")
    else:
        print("✗ GOOGLE_API_KEY not found in environment")
        print("  Please set it in Streamlit secrets or environment variables")
        all_keys_found = False
    
    # Check Pinecone API key
    if "PINECONE_API_KEY" in os.environ:
        print("✓ PINECONE_API_KEY found in environment")
    else:
        print("✗ PINECONE_API_KEY not found in environment")
        print("  Please set it in Streamlit secrets or environment variables")
        all_keys_found = False
    
    return all_keys_found

def test_pinecone_connectivity():
    """Test if Pinecone is properly configured and accessible."""
    print("\n=== Testing Pinecone Connectivity ===")
    
    try:
        from pinecone_utils import PineconeManager
        from config import Config
        
        # Initialize Pinecone manager
        manager = PineconeManager()
        
        # Check if index exists
        stats = manager.get_index_stats()
        
        if stats and 'total_vector_count' in stats:
            vector_count = stats.get('total_vector_count', 0)
            print(f"✓ Connected to Pinecone index '{Config.PINECONE_INDEX_NAME}'")
            print(f"  Total vectors: {vector_count}")
            
            if vector_count == 0:
                print("⚠️  Warning: Pinecone index is empty. Run preprocess_pdf_pinecone.py to upload vectors.")
                return True  # Connection works, just no data
            else:
                print("✓ Pinecone index contains data")
                return True
        else:
            print("✗ Could not connect to Pinecone or retrieve stats")
            return False
            
    except Exception as e:
        print(f"✗ Pinecone connectivity test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and provide a summary."""
    print("=" * 50)
    print("Running Law-GPT Test Suite")
    print("=" * 50)
    
    tests = [
        ("Preprocessed Text", test_preprocessed_text_exists),
        ("API Key Configuration", test_api_key_configuration),
        ("Pinecone Connectivity", test_pinecone_connectivity),
        ("Response Validation", test_validate_response),
        ("Input Sanitization", test_sanitize_input),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The application is ready for production.")
    else:
        print("\n✗ Some tests failed. Please fix the issues before deploying.")
    
    return passed == total

if __name__ == "__main__":
    # For testing individual functions during development
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        # Run individual tests
        test_validate_response()
        test_sanitize_input()
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)