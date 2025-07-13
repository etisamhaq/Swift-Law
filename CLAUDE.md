# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Law-GPT is a RAG-based chatbot that answers questions exclusively about Pakistan's legal system using a PDF document as its knowledge base. The application uses Streamlit for the web interface and Google's Gemini model for generating responses.

## Key Commands

### Setup and Development
```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess the PDF (MUST be run before first use)
python preprocess_pdf.py

# Run the application
streamlit run app.py

# Run tests
python test_app.py

# Production deployment
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Testing Specific Components
```bash
# Test individual functions
python test_app.py --individual
```

## Architecture and Core Components

### RAG Pipeline Flow
1. **PDF Preprocessing** (`preprocess_pdf.py`): Extracts text from Pakistan.pdf and saves to `preprocessed_text.json`
2. **Text Chunking**: Uses RecursiveCharacterTextSplitter with configurable chunk size/overlap
3. **Embeddings**: HuggingFace sentence-transformers model creates vector representations
4. **Vector Store**: FAISS stores and retrieves relevant chunks based on similarity
5. **LLM Integration**: Google Gemini model generates responses using retrieved context

### Key Functions in app.py

- **`is_law_related_question()`**: Filters non-legal queries using keyword matching
- **`validate_response()`**: Ensures responses are context-based, not from external knowledge
- **`sanitize_input()`**: Prevents injection attacks by cleaning user input
- **`create_qa_system()`**: Initializes the RAG pipeline with caching for performance

### Configuration System

The application uses a flexible configuration system:
- Primary: `config.py` reads from environment variables
- Fallback: Streamlit secrets
- Default: Hardcoded values in Config class

Key configurations:
- `GOOGLE_API_KEY`: Required for Gemini model
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Controls text splitting
- `RETRIEVAL_K` / `SCORE_THRESHOLD`: Controls context retrieval
- `MAX_INPUT_LENGTH`: Security limit on user input

### Context Enforcement Strategy

The chatbot enforces context-only responses through:
1. Custom prompt template that explicitly instructs to use only provided context
2. Law-related question detection before processing
3. Response validation to catch external knowledge indicators
4. Similarity score threshold to ensure relevant context retrieval

### Error Handling and Logging

- Comprehensive try-catch blocks for all external operations
- Configurable logging levels via LOG_LEVEL environment variable
- User-friendly error messages while logging technical details

## Important Constraints

1. **Context-Only Responses**: The chatbot MUST only answer from the Pakistan.pdf content
2. **Law Domain Only**: Non-law questions are automatically rejected
3. **Input Sanitization**: All user inputs must pass through `sanitize_input()`
4. **API Key Security**: Never commit API keys; use environment variables or Streamlit secrets

## Testing Considerations

The test suite (`test_app.py`) validates:
- Law-related question detection accuracy
- Response validation logic
- Input sanitization effectiveness
- Preprocessed text file existence
- API key configuration

Always run tests after making changes to core functions.

## Performance Optimizations

- `@st.cache_resource`: Caches the QA system initialization
- `@lru_cache`: Caches preprocessed text loading
- Streamlit's built-in caching for embeddings
- Configurable chunk sizes for memory/speed tradeoffs

## Deployment Notes

- Requires `preprocessed_text.json` to be generated before deployment
- PM2 configuration available in `ecosystem.config.js`
- Docker deployment supported (see README_PRODUCTION.md)
- Streamlit Cloud deployment ready with secrets management