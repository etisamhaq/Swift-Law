import os
import PyPDF2
import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Law-GPT: Pakistan Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

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
    
    .preset-question {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .preset-question:hover {
        background-color: #d1e7ff;
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

    # Preset questions section
    st.markdown("### 💡 Quick Start Questions")
    preset_questions = [
        "What are the fundamental rights outlined in the Constitution of Pakistan?",
        "Explain the structure of the judicial system in Pakistan.",
        "What is the process of constitutional amendment in Pakistan?"
    ]

    # Initialize chat history if not already done
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display preset questions with clickable functionality
    for question in preset_questions:
        if st.button(question, key=question, use_container_width=True):
            # Check if the question is already in the chat history
            if {"role": "user", "content": question} not in st.session_state.messages:
                # Trigger the QA system with the selected question
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Generate response
                with st.spinner('🔬 Analyzing legal documents...'):
                    try:
                        response = st.session_state.qa_system.run(question)
                    except Exception as e:
                        response = f"Apologies, an error occurred: {str(e)}"
                    
                    # Display response with assistant styling
                    st.markdown(f'<div class="assistant-message">🤖 {response}</div>', unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("You've already asked that question. Please try a different one.")

    # Initialize session state for QA system if not already done
    if 'qa_system' not in st.session_state:
        with st.spinner('🔍 Initializing the legal research assistant...'):
            st.session_state.qa_system = create_qa_system()
        st.success('🚀 Chatbot is ready to assist you!')

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
                response = st.session_state.qa_system.run(prompt)
            except Exception as e:
                response = f"Apologies, an error occurred: {str(e)}"
            
            # Display response with assistant styling
            st.markdown(f'<div class="assistant-message">🤖 {response}</div>', unsafe_allow_html=True)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown("*Powered by AI | Designed for Legal Research*")

if __name__ == "__main__":
    main()