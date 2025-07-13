import os
import streamlit as st
from functools import lru_cache
import time
import json
import logging
from typing import Optional, Dict, Any
import numpy as np

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Law-GPT: Pakistan Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone_utils import PineconeManager

# Import configuration
try:
    from config import Config
    Config.validate()
except ImportError:
    # Fallback configuration if config.py doesn't exist
    class Config:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
        MODEL_NAME = 'gemini-1.5-flash'
        TEMPERATURE = 0.1
        MAX_OUTPUT_TOKENS = 1024
        TOP_P = 0.8
        TOP_K = 40
        EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        RETRIEVAL_K = 5
        SCORE_THRESHOLD = 0.7
        MAX_INPUT_LENGTH = 500
        LOG_LEVEL = 'INFO'
        PINECONE_INDEX_NAME = 'pakistan-law'

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    
    .title {
        font-size: 2.5rem;
        color: #1a3c5f;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .user-message {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .assistant-message {
        background-color: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .stSpinner > div {
        border-color: #1a3c5f !important;
        border-top-color: transparent !important;
    }
    
    .chat-history-item {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .chat-history-item:hover {
        background-color: #e6f2ff;
    }
</style>
""", unsafe_allow_html=True)

# Set API keys from configuration
try:
    # Google API Key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif Config.GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
    elif "GOOGLE_API_KEY" not in os.environ:
        st.error("Google API key not found. Please set it in Streamlit secrets, environment variables, or config.")
        st.stop()
    
    # Pinecone API Key
    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    elif Config.PINECONE_API_KEY:
        os.environ["PINECONE_API_KEY"] = Config.PINECONE_API_KEY
    elif "PINECONE_API_KEY" not in os.environ:
        st.error("Pinecone API key not found. Please set it in Streamlit secrets, environment variables, or config.")
        st.stop()
        
except Exception as e:
    logger.error(f"Error loading API keys: {e}")
    st.error("Error loading API keys. Please check your configuration.")
    st.stop()

# Cache the PDF text extraction
@lru_cache(maxsize=1)
def load_preprocessed_text():
    try:
        with open('preprocessed_text.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("text", "")
            if not text:
                st.error("Preprocessed text file is empty.")
                return None
            logger.info(f"Loaded preprocessed text with {len(text)} characters")
            return text
    except FileNotFoundError:
        st.error("Preprocessed text file not found. Please run preprocess_pdf.py first.")
        logger.error("preprocessed_text.json not found")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error parsing preprocessed text file: {e}")
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading preprocessed text: {e}")
        logger.error(f"Unexpected error: {e}")
        return None

# Cache the QA system creation
@st.cache_resource(show_spinner=False)
def create_qa_system():
    try:
        # Initialize Pinecone manager
        logger.info("Initializing Pinecone connection...")
        pinecone_manager = PineconeManager()
        
        # Get vector store
        vector_store = pinecone_manager.get_vector_store()
        
        # Check if index has vectors
        stats = pinecone_manager.get_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            st.error("No vectors found in Pinecone index. Please run preprocess_pdf_pinecone.py first.")
            logger.error("Pinecone index is empty")
            return None
        
        logger.info(f"Connected to Pinecone index with {total_vectors} vectors")

        llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,  # Lower temperature for more consistent answers
            max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            top_p=Config.TOP_P,
            top_k=Config.TOP_K
        )
        
        # Create a custom prompt template that enforces staying within context
        prompt_template = """
        You are Law-GPT, a specialized legal assistant trained exclusively on Pakistan's legal documents.
        
        Context: {context}
        
        Question: {question}
        
        INSTRUCTIONS:
        1. ONLY answer if you can find relevant information in the provided context.
        2. If the context doesn't contain relevant information, respond with: "I don't have sufficient information about this topic in my knowledge base."
        3. When answering, quote or reference specific sections from the context.
        4. Do NOT use any external knowledge - only information from the context provided.
        5. Be precise and factual in your responses.
        6. Focus on Pakistani law and legal matters based on the provided context.
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create custom retriever with Pinecone
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": Config.RETRIEVAL_K  # Get top K most relevant chunks
            }
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True  # Enable returning source documents
        )

        return qa
    
    except Exception as e:
        logger.error(f"Error creating QA system: {e}")
        st.error(f"Failed to initialize QA system: {str(e)}")
        return None

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove any potential harmful characters or patterns
    import re
    # Remove multiple spaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove any special characters that might be used for injection
    text = re.sub(r'[<>{}\\]', '', text)
    # Limit length to prevent DoS
    max_length = Config.MAX_INPUT_LENGTH
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip()


def validate_response(response: str, source_documents: list) -> str:
    """Validate and potentially modify the response based on context relevance."""
    # Check for common indicators that the model is using external knowledge
    external_knowledge_indicators = [
        "based on general knowledge",
        "in general,",
        "typically,",
        "usually,",
        "commonly,",
        "it is known that"
    ]
    
    response_lower = response.lower()
    
    # If response contains indicators of external knowledge, return appropriate message
    if any(indicator in response_lower for indicator in external_knowledge_indicators):
        return "I don't have sufficient information about this specific topic in my knowledge base. Please note that I can only provide information from Pakistan's legal documents that I have been trained on."
    
    # Check if response is too short or generic
    if len(response.strip()) < 50:
        return "I couldn't find specific information about this in my legal knowledge base. Please try rephrasing your question or ask about a different aspect of Pakistani law."
    
    return response

def main():
    # Custom title with markdown and icon
    st.markdown('<h1 class="title">⚖️ Law-GPT: Pakistan Legal Assistant</h1>', unsafe_allow_html=True)

    # Sidebar with additional information
    st.sidebar.header("About This Assistant")
    st.sidebar.info(
        "Law-GPT is an AI-powered chatbot that provides insights "
        "into Pakistan's Constitution and Legal System. "
        "Ask questions and get precise, context-aware answers!"
    )

    # Initialize QA system with progress tracking
    if 'qa_system' not in st.session_state:
        progress_bar = st.progress(0)
        with st.spinner('🔍 Initializing the legal research assistant...'):
            start_time = time.time()
            st.session_state.qa_system = create_qa_system()
            progress_bar.progress(100)
            load_time = time.time() - start_time
            st.success(f'🚀 Chatbot ready in {load_time:.2f} seconds!')
        progress_bar.empty()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history in the sidebar
    st.sidebar.header("Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        if st.sidebar.button(f"Chat {i+1}", key=f"chat_history_item_{i}"):
            st.session_state.messages = chat

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Limit chat history size
    max_history_length = 10
    if len(st.session_state.messages) > max_history_length:
        st.session_state.messages = st.session_state.messages[-max_history_length:]

    # Display chat history with enhanced styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    # Chat input with placeholder and icon
    if prompt := st.chat_input("Ask a question about Pakistan's legal system... 💬"):
        # Sanitize input
        prompt = sanitize_input(prompt)
        
        # Check for empty input after sanitization
        if not prompt:
            st.warning("Please enter a valid question.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">👤 {prompt}</div>', unsafe_allow_html=True)

        
        # Generate response
        with st.spinner('🔬 Analyzing legal documents...'):
            try:
                # The qa_system.run now returns both the answer and source documents
                result = st.session_state.qa_system({"query": prompt})
                response = result["result"]
                
                # Get source documents for validation
                source_documents = result.get("source_documents", [])
                
                # If no relevant documents found, provide appropriate response
                if not source_documents:
                    response = "I couldn't find relevant information about this topic in my Pakistani law knowledge base. Please try asking about specific laws, acts, or constitutional matters."
                else:
                    # Validate the response
                    response = validate_response(response, source_documents)
                    
                    # Add source information if we have a valid response
                    if not response.startswith("I don't have") and not response.startswith("I can only"):
                        # Extract relevant chunks from source documents
                        relevant_chunks = []
                        for doc in source_documents[:3]:  # Show top 3 sources
                            if hasattr(doc, 'page_content'):
                                chunk = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                # Add chunk metadata if available
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                                    chunk = f"[Chunk {chunk_id}] {chunk}"
                                relevant_chunks.append(chunk)
                        
                        if relevant_chunks:
                            response += "\n\n📚 **Referenced Sections:**\n"
                            for i, chunk in enumerate(relevant_chunks, 1):
                                response += f"\n{i}. {chunk}"
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                response = "An error occurred while processing your question. Please try again or rephrase your question."
            
            # Display response with assistant styling
            st.markdown(f'<div class="assistant-message">🤖 {response}</div>', unsafe_allow_html=True)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update the chat history
        st.session_state.chat_history.append(st.session_state.messages.copy())

        # Maintain a maximum of 3 chat history items
        if len(st.session_state.chat_history) > 3:
            st.session_state.chat_history.pop(0)

    # Debug toggle in sidebar to show knowledge base statistics
    with st.sidebar.expander("Debug Information"):
        if st.button("Show Knowledge Base Stats"):
            if 'qa_system' in st.session_state:
                try:
                    pinecone_manager = PineconeManager()
                    stats = pinecone_manager.get_index_stats()
                    st.sidebar.write(f"**Pinecone Index Stats:**")
                    st.sidebar.write(f"Total vectors: {stats.get('total_vector_count', 0)}")
                    st.sidebar.write(f"Index name: {Config.PINECONE_INDEX_NAME}")
                    st.sidebar.write(f"Dimension: {stats.get('dimension', Config.PINECONE_DIMENSION)}")
                    st.sidebar.write(f"Index fullness: {stats.get('index_fullness', 0):.2%}")
                except Exception as e:
                    st.sidebar.write(f"Could not retrieve stats: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by AI | Designed for Legal Research*")

if __name__ == "__main__":
    main()
