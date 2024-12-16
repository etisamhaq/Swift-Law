# Law-GPT Chatbot

## Overview

The Law-GPT Chatbot is a web application built using Streamlit that allows users to ask questions about Pakistan's Constitution and Legal System. The application utilizes LangChain and Google Generative AI to provide accurate and relevant answers.

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
   git clone https://github.com/etisamhaq/lawgpt-streamlit.git
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

## File Structure

law-gpt-chatbot/
│
├── app.py # Main application file
├── config.py # Configuration file for API keys
├── requirements.txt # List of required Python packages
├── .gitignore # Files and directories to ignore in Git
└── Pakistan.pdf # PDF document containing the Constitution of Pakistan


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an easy way to build web applications.
- [LangChain](https://langchain.readthedocs.io/en/latest/) for enabling the integration of language models.
- [Google Generative AI](https://cloud.google.com/generative-ai) for providing powerful AI capabilities.

## Contact

For any inquiries or feedback, please reach out to [your.email@example.com](mailto:your.email@example.com).
