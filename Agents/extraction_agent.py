import os
import pytesseract
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from docx import Document  # NEW: For handling .docx files
from models.state import GraphState
from config import TESSERACT_CMD

# Configure Tesseract path
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- HELPER FUNCTIONS ---

def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")

def is_image(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))

def is_docx(path: str) -> bool:
    # Note: .doc (binary) is harder to support; .docx (XML) is standard
    return path.lower().endswith(".docx")

def ocr_image(image_input) -> str:
    """Helper to run OCR on an image object or path with error handling."""
    try:
        # Tesseract works better on grayscale images
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('L') # Convert to grayscale
        else:
            img = image_input.convert('L')
            
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

# --- MAIN EXTRACTION LOGIC ---

def extract_from_files(state: GraphState) -> GraphState:
    documents = []
    
    if not state.get("files"):
        return {**state, "documents": []}
    
    for file_path in state["files"]:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        extracted_text = ""
        
        try:
            # 1. HANDLE PDF (Hybrid: Text Layer -> Fallback to OCR)
            if is_pdf(file_path):
                print(f"Processing PDF: {filename}...")
                
                # Attempt 1: Fast text extraction
                try:
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                except Exception as e:
                    print(f"PyPDF failed on {filename}: {e}")

                # Attempt 2: OCR Fallback (if text is empty or too short)
                if len(extracted_text.strip()) < 50:
                    print(f"PDF text layer empty/sparse. Switching to OCR for {filename}...")
                    # Convert PDF pages to images
                    # Note: You might need poppler_path in config if on Windows
                    images = convert_from_path(file_path) 
                    extracted_text = "" # Reset
                    for img in images:
                        extracted_text += ocr_image(img) + "\n"

            # 2. HANDLE IMAGES (JPG, PNG, etc.)
            elif is_image(file_path):
                print(f"Processing Image: {filename}...")
                extracted_text = ocr_image(file_path)

            # 3. HANDLE WORD DOCS (.docx)
            elif is_docx(file_path):
                print(f"Processing DOCX: {filename}...")
                doc = Document(file_path)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])

            else:
                print(f"Warning: Unsupported file type: {file_path}")
                continue

            # Final check before adding
            if extracted_text.strip():
                documents.append({
                    "doc_id": filename,
                    "text": extracted_text.strip()
                })
            else:
                print(f"Warning: No text could be extracted from {filename}")

        except Exception as e:
            print(f"Critical error extracting {filename}: {e}")

    return {
        **state,
        "documents": documents
    }

# Keep your routing/skip logic as is
def route_extraction(state: GraphState) -> str:
    if state.get("files"):
        return "extract"
    return "skip"

def skip_extraction(state: GraphState) -> GraphState:
    return {**state, "documents": []}