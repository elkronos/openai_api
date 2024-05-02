import io
import os
import re
import requests
from pdf2image import convert_from_path
from docx import Document
import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Ensure Tesseract-OCR is installed and set the path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as necessary

# API key should be set in your environment variables for security reasons
api_key = os.getenv('OPENAI_API_KEY', 'sk-your-key-here')  # Replace 'sk-your-key-here' with your actual API key or set it in your environment variables

def setup_session():
    """Set up session with retry logic."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def ocr_image(image):
    """Perform OCR on an image."""
    return pytesseract.image_to_string(image)

def parse_text(file_path, remove_whitespace=True, remove_special_chars=True, remove_numbers=True):
    """Parse documents to extract text, handling text-based and scanned PDFs, DOCX, and TXT files.
    
    Args:
        file_path (str): Path to the file.
        remove_whitespace (bool): Flag to remove excessive whitespace.
        remove_special_chars (bool): Flag to remove special characters.
        remove_numbers (bool): Flag to remove numbers.
    
    Returns:
        list: Chunks of processed text, each limited to 15,000 tokens.
        
    Raises:
        ValueError: If the file extension is unsupported.
        IOError: If there is an error reading the file.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == '.pdf':
            doc = fitz.open(file_path)
            if any(not page.is_text_page() for page in doc):
                images = convert_from_path(file_path)
                text = ' '.join(ocr_image(image) for image in images)
            else:
                text = ''.join(page.get_text("text") for page in doc if page.is_text_page())
        elif ext == '.docx':
            doc = Document(file_path)
            text = ' '.join(para.text for para in doc.paragraphs)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise IOError(f"Failed to read or process the file {file_path}: {e}")

    # Text cleaning steps
    if remove_whitespace:
        text = ' '.join(text.split())
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Splitting text into manageable chunks for processing
    tokens = text.split()
    token_chunks = [' '.join(tokens[i:i + 15000]) for i in range(0, len(tokens), 15000)]

    return token_chunks


def gpt_read(chunk_list, question, model="gpt-3.5-turbo-1106", temperature=0.0, max_tokens=16000):
    """Send chunks of text to the GPT model to generate responses."""
    session = setup_session()
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    url = "https://api.openai.com/v1/chat/completions"

    responses = []
    for chunk in chunk_list:
        data = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a research assistant tasked to analyze the document."},
                {"role": "user", "content": chunk},
                {"role": "user", "content": question}
            ]
        }
        
        response = session.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if result['choices']:
            responses.append(result['choices'][0]['message']['content'])
    
    final_response = ' '.join(responses)
    return final_response

# Example usage:
file_path = "path/to/document.pdf"
text_chunks = parse_text(file_path)
question = "What is the main topic of the document?"
response = gpt_read(text_chunks, question)
print(response)
