import PyPDF2
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_utils import PineconeManager
from config import Config
import sys

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def preprocess_and_upload_to_pinecone(pdf_path: str = "Pakistan.pdf", clear_existing: bool = True):
    """
    Preprocess PDF and upload embeddings to Pinecone.
    
    Args:
        pdf_path: Path to the PDF file
        clear_existing: Whether to clear existing vectors before uploading
    """
    try:
        # Initialize Pinecone manager
        logger.info("Initializing Pinecone manager...")
        pinecone_manager = PineconeManager()
        
        # Create index if it doesn't exist
        logger.info("Creating Pinecone index if needed...")
        pinecone_manager.create_index_if_not_exists()
        
        # Clear existing vectors if requested
        if clear_existing:
            logger.info("Clearing existing vectors...")
            pinecone_manager.delete_all_vectors()
        
        # Extract text from PDF
        logger.info(f"Extracting text from {pdf_path}...")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{total_pages} pages...")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Split text into chunks
        logger.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", ",", " ", ""]
        )
        texts = text_splitter.split_text(text)
        logger.info(f"Created {len(texts)} text chunks")
        
        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(texts):
            metadata = {
                "source": pdf_path,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(texts)
            }
            metadatas.append(metadata)
        
        # Upload to Pinecone in batches
        logger.info("Uploading chunks to Pinecone...")
        batch_size = Config.PINECONE_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            pinecone_manager.upload_documents(batch_texts, batch_metadatas)
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Save preprocessed text locally as backup
        logger.info("Saving preprocessed text locally...")
        with open('preprocessed_text.json', 'w', encoding='utf-8') as f:
            json.dump({
                "text": text,
                "chunks": texts,
                "total_chunks": len(texts),
                "total_characters": len(text)
            }, f, ensure_ascii=False)
        
        # Get index statistics
        stats = pinecone_manager.get_index_stats()
        logger.info(f"Pinecone index stats: {stats}")
        
        logger.info("✅ Successfully preprocessed PDF and uploaded to Pinecone!")
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing PDF: {e}")
        return False

if __name__ == "__main__":
    # Check if clear flag is provided
    clear_existing = "--clear" in sys.argv
    
    if clear_existing:
        response = input("⚠️  This will delete all existing vectors in Pinecone. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            sys.exit(0)
    
    success = preprocess_and_upload_to_pinecone(clear_existing=clear_existing)
    sys.exit(0 if success else 1)