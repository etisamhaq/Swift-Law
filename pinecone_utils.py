import logging
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import Config

logger = logging.getLogger(__name__)

class PineconeManager:
    """Manages Pinecone operations for the Law-GPT application."""
    
    def __init__(self):
        """Initialize Pinecone client and embeddings."""
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=Config.PINECONE_DIMENSION,
                    metric=Config.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud='gcp',
                        region='us-central1'
                    )
                )
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating/checking index: {e}")
            raise
    
    def get_index(self):
        """Get Pinecone index."""
        return self.pc.Index(self.index_name)
    
    def delete_all_vectors(self):
        """Delete all vectors from the index."""
        try:
            index = self.get_index()
            index.delete(delete_all=True)
            logger.info("Deleted all vectors from index")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def get_vector_store(self) -> PineconeVectorStore:
        """Get Pinecone vector store for LangChain integration."""
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=Config.PINECONE_API_KEY
        )
    
    def upload_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Upload documents to Pinecone."""
        try:
            vector_store = self.get_vector_store()
            
            # If no metadata provided, create default metadata
            if metadatas is None:
                metadatas = [{"source": "Pakistan.pdf", "chunk_id": i} for i in range(len(texts))]
            
            # Add texts to vector store
            vector_store.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Successfully uploaded {len(texts)} documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search on Pinecone index."""
        try:
            vector_store = self.get_vector_store()
            
            # Perform similarity search with score
            results = vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered_results = []
            for doc, score in results:
                if score >= score_threshold:
                    filtered_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score
                    })
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            index = self.get_index()
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}