# Law-GPT: Production Deployment Guide

This document provides instructions for deploying the Law-GPT Pakistan Legal Assistant in a production environment.

## Overview

Law-GPT is a RAG-based chatbot that answers questions about Pakistan's legal system. The improvements made include:

### Key Fixes Implemented:
1. **Context-Only Responses**: The chatbot now strictly answers only from the provided legal context
2. **Non-Law Question Filtering**: Automatically detects and rejects non-law related questions
3. **Improved Context Retrieval**: Better chunking and retrieval with relevance scoring
4. **Input Validation**: Sanitizes user input to prevent injection attacks
5. **Error Handling**: Comprehensive error handling and logging
6. **Production Configuration**: Environment-based configuration system
7. **Performance Optimization**: Caching and optimized embeddings

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini model
- Pakistan.pdf file (legal document)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd Swift-Law
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Preprocess the PDF** (if not already done):
   ```bash
   python preprocess_pdf.py
   ```

## Configuration

Edit the `.env` file with your production settings:

```env
# Required
GOOGLE_API_KEY=your_actual_api_key_here

# Optional (defaults shown)
APP_ENV=production
LOG_LEVEL=INFO
MODEL_NAME=gemini-1.5-flash
TEMPERATURE=0.1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
SCORE_THRESHOLD=0.7
MAX_INPUT_LENGTH=500
```

## Running the Application

### Development Mode:
```bash
streamlit run app.py
```

### Production Mode:
```bash
# Using streamlit directly
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Or using a process manager like PM2
pm2 start ecosystem.config.js
```

## Testing

Run the test suite to verify everything is working:

```bash
python test_app.py
```

Expected output:
- All tests should pass
- The test suite checks:
  - Law-related question detection
  - Response validation
  - Input sanitization
  - Preprocessed text availability
  - API key configuration

## Production Deployment Options

### 1. Streamlit Cloud (Recommended for Quick Deployment)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repository
4. Add GOOGLE_API_KEY in Streamlit secrets

### 2. Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t law-gpt .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key law-gpt
```

### 3. Cloud Platform Deployment (AWS/GCP/Azure)
- Use platform-specific services (App Engine, EC2, Cloud Run, etc.)
- Ensure you set environment variables
- Configure proper security groups/firewall rules

## Security Considerations

1. **API Key Security**: Never commit API keys to version control
2. **Input Validation**: The app sanitizes all user inputs
3. **Rate Limiting**: Consider adding rate limiting for production
4. **HTTPS**: Always use HTTPS in production
5. **Authentication**: Consider adding authentication for sensitive deployments

## Monitoring

1. **Logging**: Check logs at the configured LOG_LEVEL
2. **Performance**: Monitor response times and memory usage
3. **Errors**: Track error rates and types
4. **Usage**: Monitor API usage to stay within limits

## Troubleshooting

### Common Issues:

1. **"Preprocessed text file not found"**
   - Run: `python preprocess_pdf.py`

2. **"Google API key not found"**
   - Check your .env file or environment variables
   - Ensure the key is valid

3. **"No module named 'langchain'"**
   - Run: `pip install -r requirements.txt`

4. **Slow responses**
   - Check your internet connection
   - Consider increasing CHUNK_SIZE and decreasing RETRIEVAL_K

5. **Out of memory errors**
   - Reduce CHUNK_SIZE
   - Use a smaller embedding model

## Performance Tips

1. The app caches the QA system for faster subsequent loads
2. Embeddings are cached using Streamlit's cache mechanism
3. Consider using a GPU for embeddings if available
4. Adjust chunk size based on your document size

## Support

For issues or questions:
1. Check the test output: `python test_app.py`
2. Review logs for error messages
3. Ensure all dependencies are correctly installed
4. Verify your Google API key is valid and has necessary permissions

## License

[Your License Here]