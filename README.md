# Law-GPT Chatbot

## Overview

The Law-GPT Chatbot is a web application built using Streamlit that allows users to ask questions about Pakistan's Constitution and Legal System. The application utilizes LangChain and Google Generative AI to provide accurate and relevant answers based on the content of a PDF document, specifically "Pakistan.pdf".

## Features

- **Interactive Chat Interface**: Users can interact with the chatbot in real-time.
- **PDF Text Extraction**: The application extracts text from a PDF document to provide contextually relevant answers.
- **Natural Language Processing**: Utilizes advanced AI models to understand and respond to user queries.
- **Session Management**: Maintains chat history for a seamless user experience.

## Technologies Used

- **Python**: The primary programming language for the application.
- **Streamlit**: A framework for building web applications in Python.
- **LangChain**: A framework for building applications with language models.
- **Google Generative AI**: For generating responses based on user queries.
- **PyPDF2**: For extracting text from PDF files.
- **Chroma**: A vector store for managing embeddings.
- **HuggingFaceEmbeddings**: For creating embeddings from text.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/law-gpt-chatbot.git
   cd law-gpt-chatbot
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google API key**:
   - Create a `config.py` file in the root directory and add your Google API key:
     ```python
     GOOGLE_API_KEY = 'your_google_api_key_here'
     ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. You will see the chatbot interface.
3. Type your question regarding Pakistan's Constitution or Legal System in the chat input box.
4. The chatbot will respond with relevant information extracted from the PDF document.

## File Structure
