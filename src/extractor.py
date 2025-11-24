"""PDF text extraction module."""

import logging
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    try:
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {file_path}")
            return None
            
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
                logger.info(f"Extracted text from page {page_num}/{len(reader.pages)}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path.name}")
        
        return full_text if full_text else None
        
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace and normalizing.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = [line.strip() for line in text.split('\n')]
    # Remove empty lines
    lines = [line for line in lines if line]
    # Join with single newline
    cleaned = '\n'.join(lines)
    
    return cleaned
