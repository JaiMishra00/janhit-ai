"""
Agent 1: Document Extraction
Extracts text from PDFs and images using PyPDF and Tesseract OCR.
"""

import os
import pytesseract
from pypdf import PdfReader
from PIL import Image
from models.state import GraphState
from config import TESSERACT_CMD

# Configure Tesseract path
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def is_pdf(path: str) -> bool:
    """Check if file is a PDF."""
    return path.lower().endswith(".pdf")


def is_image(path: str) -> bool:
    """Check if file is an image."""
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))


def extract_from_files(state: GraphState) -> GraphState:
    """
    Extract text from PDF and image files.
    
    State inputs:
        - files: List of file paths
        
    State outputs:
        - documents: List of {"doc_id": str, "text": str}
        
    Returns:
        Updated state with extracted documents
    """
    documents = []
    
    # If no files provided, return empty documents list
    if not state.get("files"):
        return {
            **state,
            "documents": []
        }
    
    for file_path in state["files"]:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        
        # ========== PDF EXTRACTION ==========
        if is_pdf(file_path):
            try:
                reader = PdfReader(file_path)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                documents.append({
                    "doc_id": filename,
                    "text": text.strip()
                })
            except Exception as e:
                print(f"Error extracting PDF {filename}: {e}")
        
        # ========== IMAGE OCR ==========
        elif is_image(file_path):
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                
                documents.append({
                    "doc_id": filename,
                    "text": text.strip()
                })
            except Exception as e:
                print(f"Error performing OCR on {filename}: {e}")
        
        else:
            print(f"Warning: Unsupported file type: {file_path}")
    
    return {
        **state,
        "documents": documents
    }


def route_extraction(state: GraphState) -> str:
    """
    Route decision: determine if extraction is needed.
    
    Returns:
        "extract" if files are present, "skip" otherwise
    """
    if state.get("files"):
        return "extract"
    return "skip"


def skip_extraction(state: GraphState) -> GraphState:
    """
    Pass-through for text-only queries.
    
    Returns:
        State with empty documents list
    """
    return {
        **state,
        "documents": []
    }