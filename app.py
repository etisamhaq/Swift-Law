import os
import PyPDF2
import streamlit as st
from functools import lru_cache
import time
import json

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Law-GPT: Pakistan Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

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

# Set API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Cache the PDF text extraction
@lru_cache(maxsize=1)
def load_preprocessed_text():
    try:
        with open('preprocessed_text.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["text"]
    except FileNotFoundError:
        st.error("Preprocessed text file not found. Please run preprocess_pdf.py first.")
        return None

# Cache the QA system creation
@lru_cache(maxsize=1)
def create_qa_system():
    # Load preprocessed text instead of processing PDF
    text = load_preprocessed_text()
    if text is None:
        return None

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    
    db = FAISS.from_texts(texts, embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
    )
    
    # Create a custom prompt template that enforces staying within context
    prompt_template = """
    You are Law-GPT, a specialized assistant for Pakistan's legal system. Answer ONLY based on the context provided.
    
    Context: {context}
    
    Question: {question}
    
    Important instructions:
    1. If the context doesn't contain information to answer the question, respond with: "I don't have enough information in my knowledge base to answer this question about Pakistan's legal system."
    2. Don't use knowledge outside of the provided context.
    3. Don't make up or infer information not present in the context.
    4. Be precise and cite relevant sections from the context when possible.
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(
            search_kwargs={
                "k": 3,  # Increased from 1 to get more context
                "score_threshold": 0.5  # Increased threshold for higher relevance
            }
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Enable returning source documents
    )

    return qa

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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">👤 {prompt}</div>', unsafe_allow_html=True)

        # Generate response
        with st.spinner('🔬 Analyzing legal documents...'):
            try:
                # The qa_system.run now returns both the answer and source documents
                result = st.session_state.qa_system({"query": prompt})
                response = result["result"]
                
                # Add sources information if available
                if hasattr(result, "source_documents") and result["source_documents"]:
                    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"] if hasattr(doc, "metadata")]
                    if sources:
                        response += f"\n\nSources: {', '.join(set(sources))}"
                
                # Check if the response indicates no information was found
                if "don't have enough information" in response.lower() or "insufficient information" in response.lower():
                    response = "I don't have enough information in my knowledge base to answer this question about Pakistan's legal system. My responses are limited to the specific legal documents I've been trained on."
            except Exception as e:
                response = f"Apologies, an error occurred: {str(e)}"
            
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
                    db = st.session_state.qa_system.retriever.vectorstore
                    st.sidebar.write(f"Total documents: {len(db.index_to_docstore_id)}")
                    st.sidebar.write(f"Embedding dimensions: {db.embedding_dim}")
                except:
                    st.sidebar.write("Could not retrieve stats")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by AI | Designed for Legal Research*")

if __name__ == "__main__":
    main()
