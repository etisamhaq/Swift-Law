#!/usr/bin/env python3
"""
Migration script to move from local preprocessed_text.json to Pinecone vector database.
This script reads the existing preprocessed text and uploads it to Pinecone.
"""

import json
import logging
import sys
from preprocess_pdf_pinecone import preprocess_and_upload_to_pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_to_pinecone():
    """Migrate existing preprocessed text to Pinecone."""
    try:
        # Check if preprocessed_text.json exists
        logger.info("Checking for existing preprocessed_text.json...")
        try:
            with open('preprocessed_text.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Found preprocessed text with {len(data.get('text', ''))} characters")
        except FileNotFoundError:
            logger.info("No existing preprocessed_text.json found.")
            response = input("Would you like to process Pakistan.pdf directly? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Migration cancelled.")
                return False
        
        # Ask user for confirmation
        print("\n" + "="*50)
        print("MIGRATION TO PINECONE")
        print("="*50)
        print("\nThis script will:")
        print("1. Create a Pinecone index if it doesn't exist")
        print("2. Process Pakistan.pdf and create embeddings")
        print("3. Upload all embeddings to Pinecone")
        print("\n⚠️  This will REPLACE any existing data in Pinecone!")
        
        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Migration cancelled.")
            return False
        
        # Run the preprocessing and upload
        logger.info("Starting migration to Pinecone...")
        success = preprocess_and_upload_to_pinecone(clear_existing=True)
        
        if success:
            logger.info("✅ Migration completed successfully!")
            print("\n" + "="*50)
            print("NEXT STEPS:")
            print("="*50)
            print("1. Update your .env file with PINECONE_API_KEY")
            print("2. The app will now use Pinecone for vector search")
            print("3. You can delete the local preprocessed_text.json if desired")
            print("4. Test the application to ensure everything works")
        else:
            logger.error("❌ Migration failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False

if __name__ == "__main__":
    success = migrate_to_pinecone()
    sys.exit(0 if success else 1)