

import io
import PyPDF2
import requests


# Function to get the contents of a PDF from a URL
def read_pdf_from_url(url: str) -> str:
    response = requests.get(url, verify=False)
    content = io.BytesIO(response.content)
    reader = PyPDF2.PdfReader(content)
    text = ""
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text()
    return text