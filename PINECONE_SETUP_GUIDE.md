# Pinecone Setup Guide for Law-GPT

This guide explains how to set up Pinecone vector database for the Law-GPT application.

## Why Pinecone?

The application now uses Pinecone instead of local FAISS for:
- **Scalability**: Cloud-based vector storage that can handle millions of vectors
- **Performance**: Optimized semantic similarity search
- **Persistence**: No need to reload embeddings on each app restart
- **Production-Ready**: Built for production workloads

## Prerequisites

1. Python 3.8+ installed
2. Google API Key (for Gemini model)
3. Pinecone account (free tier available)

## Step 1: Create a Pinecone Account

1. Go to [pinecone.io](https://www.pinecone.io/)
2. Sign up for a free account
3. You'll get a free starter environment (usually `gcp-starter`)

## Step 2: Get Your Pinecone API Key

1. Log into your Pinecone console
2. Navigate to "API Keys" section
3. Copy your API key (it looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

## Step 3: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your keys:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key
   PINECONE_API_KEY=your_actual_pinecone_api_key
   ```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Upload Pakistan Law Document to Pinecone

### Option A: Fresh Upload (Recommended)
```bash
python preprocess_pdf_pinecone.py --clear
```

This will:
- Create a Pinecone index named `pakistan-law`
- Process the Pakistan.pdf file
- Create embeddings for all text chunks
- Upload embeddings to Pinecone

### Option B: Migrate from Existing Setup
If you have an existing `preprocessed_text.json`:
```bash
python migrate_to_pinecone.py
```

## Step 6: Verify Setup

Run the test suite to ensure everything is configured correctly:
```bash
python test_app.py
```

You should see:
- ✓ API keys found
- ✓ Pinecone connectivity successful
- ✓ Vectors present in index

## Step 7: Run the Application

```bash
streamlit run app.py
```

## Pinecone Index Details

The application creates an index with these specifications:
- **Index Name**: `pakistan-law`
- **Dimensions**: 384 (for sentence-transformers/all-MiniLM-L6-v2)
- **Metric**: Cosine similarity
- **Environment**: gcp-starter (free tier)

## Troubleshooting

### "Pinecone API key not found"
- Ensure your `.env` file contains `PINECONE_API_KEY`
- Check that the API key is valid

### "No vectors found in Pinecone index"
- Run `python preprocess_pdf_pinecone.py --clear`
- This uploads the law document to Pinecone

### "Index not found"
- The index will be created automatically when you run `preprocess_pdf_pinecone.py`
- Ensure your Pinecone API key has permission to create indexes

### Memory/Performance Issues
- Adjust `CHUNK_SIZE` in `.env` (smaller = more chunks but better granularity)
- Adjust `RETRIEVAL_K` (fewer results = faster but might miss context)

## Cost Considerations

### Free Tier Limits (as of 2024):
- 1 index
- ~100K vectors (depends on dimensions)
- 1M requests/month

The Pakistan law document typically creates ~500-1000 chunks, well within free tier limits.

## Monitoring

In the Pinecone console, you can monitor:
- Number of vectors stored
- Query latency
- Index size
- API usage

## Security Best Practices

1. **Never commit API keys**: Keep them in `.env` or environment variables
2. **Use read-only keys in production**: Create separate keys for uploading vs querying
3. **Monitor usage**: Set up alerts for unusual query patterns
4. **Backup important data**: Keep `preprocessed_text.json` as backup

## For Streamlit Cloud Deployment

Add these secrets in Streamlit Cloud settings:
```toml
GOOGLE_API_KEY = "your-google-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "pakistan-law"
```

## Advanced Configuration

Edit `.env` for fine-tuning:
- `PINECONE_BATCH_SIZE`: Number of vectors to upload at once
- `EMBEDDING_MODEL`: Change the embedding model (update dimension accordingly)
- `CHUNK_SIZE`/`CHUNK_OVERLAP`: Adjust text chunking strategy

## Next Steps

1. Test with various law-related queries
2. Monitor Pinecone dashboard for performance
3. Consider upgrading to paid tier for production use
4. Implement additional features like metadata filtering