import streamlit as st
import json
import os
from unittest.mock import patch, MagicMock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from app
from app import is_law_related_question, validate_response, sanitize_input

def test_is_law_related_question():
    """Test the law-related question detection function."""
    print("\n=== Testing Law-Related Question Detection ===")
    
    # Test cases
    test_cases = [
        # Law-related questions
        ("What is Article 25 of Pakistan Constitution?", True),
        ("Tell me about criminal law in Pakistan", True),
        ("What are the rights of citizens under Pakistani law?", True),
        ("Explain the judicial system of Pakistan", True),
        ("What is the punishment for theft in Pakistan?", True),
        
        # Non-law related questions
        ("What is the weather today?", False),
        ("How to cook biryani?", False),
        ("Tell me about machine learning", False),
        ("What is 2 + 2?", False),
        ("Who won the cricket match?", False),
    ]
    
    passed = 0
    for question, expected in test_cases:
        result = is_law_related_question(question)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{question}' -> Expected: {expected}, Got: {result}")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

def test_validate_response():
    """Test the response validation function."""
    print("\n=== Testing Response Validation ===")
    
    test_cases = [
        # Response with external knowledge indicators
        ("Based on general knowledge, the law states...", "What is the law?", True),
        ("Typically, in Pakistan law...", "Tell me about Pakistan law", True),
        
        # Short/generic responses
        ("Yes", "Is this legal?", True),
        ("No information found", "What is Article 25?", True),
        
        # Valid responses
        ("According to Article 25 of the Constitution of Pakistan, all citizens are equal before law and are entitled to equal protection of law.", "What is Article 25?", False),
        ("The Pakistan Penal Code Section 379 defines theft as whoever intending to take dishonestly any moveable property out of the possession of any person without that person's consent.", "What is theft in Pakistan law?", False),
    ]
    
    passed = 0
    for response, question, should_be_modified in test_cases:
        result = validate_response(response, question)
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
    """Test if API key is properly configured."""
    print("\n=== Testing API Key Configuration ===")
    
    # Check environment variable
    if "GOOGLE_API_KEY" in os.environ:
        print("✓ GOOGLE_API_KEY found in environment")
        return True
    else:
        print("✗ GOOGLE_API_KEY not found in environment")
        print("  Please set it in Streamlit secrets or environment variables")
        return False

def run_all_tests():
    """Run all tests and provide a summary."""
    print("=" * 50)
    print("Running Law-GPT Test Suite")
    print("=" * 50)
    
    tests = [
        ("Preprocessed Text", test_preprocessed_text_exists),
        ("API Key Configuration", test_api_key_configuration),
        ("Law-Related Question Detection", test_is_law_related_question),
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
        test_is_law_related_question()
        test_validate_response()
        test_sanitize_input()
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)