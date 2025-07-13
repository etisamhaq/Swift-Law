# Streamlit Cloud Deployment Guide for Law-GPT

## Prerequisites
Before deploying to Streamlit Cloud, ensure you have:
1. A GitHub account
2. Your project pushed to a GitHub repository
3. A Google API key for Gemini model

## Step-by-Step Deployment Instructions

### Step 1: Prepare Your Repository

1. **Ensure these files are in your repository:**
   - ✅ `app.py` (main application)
   - ✅ `requirements.txt` (dependencies)
   - ✅ `preprocessed_text.json` (MUST be committed)
   - ✅ `preprocess_pdf.py`
   - ✅ `config.py`
   - ✅ `.gitignore` (updated to exclude secrets)

2. **Commit and push all changes:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app" button
   - Select your repository: `Swift-Law`
   - Select branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

### Step 3: Configure Secrets

1. **While deployment is in progress:**
   - Click on "Advanced settings" or go to app settings after deployment
   - Navigate to "Secrets" section

2. **Add your Google API key:**
   ```toml
   GOOGLE_API_KEY = "your-actual-google-api-key-here"
   ```

3. **Optional: Add other configuration values:**
   ```toml
   # Optional configurations (defaults will be used if not set)
   MODEL_NAME = "gemini-1.5-flash"
   TEMPERATURE = 0.1
   CHUNK_SIZE = 1000
   CHUNK_OVERLAP = 200
   RETRIEVAL_K = 5
   SCORE_THRESHOLD = 0.7
   MAX_INPUT_LENGTH = 500
   LOG_LEVEL = "INFO"
   ```

4. **Save the secrets**

### Step 4: Monitor Deployment

1. **Check deployment logs:**
   - Watch for any errors during deployment
   - The app will automatically restart when you save secrets

2. **Common deployment issues:**
   - If you see "preprocessed_text.json not found", ensure the file is committed to your repository
   - If you see API key errors, double-check your secrets configuration

### Step 5: Access Your App

Once deployed successfully:
- Your app will be available at: `https://[your-app-name].streamlit.app`
- Share this URL with users who need access to the Law-GPT chatbot

## Important Notes

### Security Considerations
- ⚠️ **Never commit your API key to GitHub**
- ✅ Always use Streamlit secrets for sensitive data
- ✅ The `.gitignore` file has been updated to exclude sensitive files

### File Size Limits
- Streamlit Cloud has a 1GB limit for repository size
- The `preprocessed_text.json` file must be under this limit
- If your PDF is very large, consider optimizing the preprocessing

### Resource Limits
- Free tier: 1GB RAM, 1GB storage
- If you encounter memory issues:
  - Reduce `CHUNK_SIZE` in secrets
  - Use a smaller embedding model

### Updating Your App
After deployment, any push to your main branch will automatically trigger a redeployment:
```bash
git add .
git commit -m "Update app"
git push origin main
```

## Troubleshooting

### "Module not found" errors
- Ensure all dependencies are in `requirements.txt`
- Check that package names are correct

### "API key not found" errors
- Verify secrets are saved correctly
- Check there are no extra spaces in the key

### "Out of memory" errors
- Reduce chunk size in secrets
- Consider upgrading to a paid Streamlit Cloud plan

### App crashes on startup
- Check deployment logs for specific errors
- Ensure `preprocessed_text.json` exists and is valid
- Verify all required files are committed

## Quick Checklist

Before deploying, ensure:
- [ ] All code changes are committed and pushed
- [ ] `preprocessed_text.json` is in the repository
- [ ] `.gitignore` excludes sensitive files
- [ ] You have your Google API key ready
- [ ] No API keys or secrets are committed to the repository

## Next Steps

After successful deployment:
1. Test the chatbot with various law-related questions
2. Monitor the app logs for any errors
3. Share the app URL with your users
4. Consider setting up a custom domain (paid feature)