import PyPDF2
import json

def preprocess_pdf():
    pdf_path = "Pakistan.pdf"
    
    # Extract text from PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # Save extracted text to file
    with open('preprocessed_text.json', 'w', encoding='utf-8') as f:
        json.dump({"text": text}, f, ensure_ascii=False)

if __name__ == "__main__":
    preprocess_pdf() 