import fitz
import os

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_all_pdfs():
    folder = "data"
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")]