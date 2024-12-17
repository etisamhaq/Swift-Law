import os
import PyPDF2
import streamlit as st
from datetime import datetime

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
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: left;
    }
    
    .timestamp {
        font-size: 0.7rem;
        color: #666;
        margin-top: 5px;
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
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-history-item:hover {
        background-color: #e6f2ff;
    }
    
    .delete-chat-btn {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Set API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_qa_system():
    pdf_path = "Pakistan.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(pdf_text)

    embeddings = HuggingFaceEmbeddings()
    
    db = FAISS.from_texts(texts, embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 1}))

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

    # Initialize session state for QA system and chat history if not already done
    if 'qa_system' not in st.session_state:
        with st.spinner('🔍 Initializing the legal research assistant...'):
            st.session_state.qa_system = create_qa_system()
        st.success('🚀 Chatbot is ready to assist you!')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history management in the sidebar
    st.sidebar.header("Chat History")
    
    # New feature: Delete individual chat history sessions
    for i, chat in enumerate(st.session_state.chat_history):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(f"Chat {i+1} - {chat[0]['timestamp']}", key=f"chat_history_item_{i}"):
                st.session_state.messages = chat
        with col2:
            if st.button("❌", key=f"delete_chat_{i}"):
                st.session_state.chat_history.pop(i)
                st.experimental_rerun()

    # Clear all chat history button
    if st.sidebar.button("🗑️ Clear All Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.experimental_rerun()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history with enhanced styling and timestamps
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
                <div class="user-message">
                    👤 {message["content"]}
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="assistant-message">
                    🤖 {message["content"]}
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            ''', unsafe_allow_html=True)

    # Chat input with placeholder and icon
    if prompt := st.chat_input("Ask a question about Pakistan's legal system... 💬"):
        # Get current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add user message to chat history with timestamp
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": current_timestamp
        })

        # Generate response
        with st.spinner('🔬 Analyzing legal documents...'):
            try:
                response = st.session_state.qa_system.run(prompt)
            except Exception as e:
                response = f"Apologies, an error occurred: {str(e)}"
            
        # Add assistant response to chat history with timestamp
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": current_timestamp
        })

        # Add current session to chat history
        st.session_state.chat_history.append(st.session_state.messages.copy())

        # Maintain a maximum of 5 chat history items
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history.pop(0)

        # Rerun to update the display
        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("*Powered by AI | Designed for Legal Research*")

if __name__ == "__main__":
    main()