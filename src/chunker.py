"""Text chunking module for splitting documents into manageable pieces."""

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks while preserving context.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    # Use RecursiveCharacterTextSplitter for smart splitting
    # It tries to split on paragraphs, then sentences, then words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    
    # Filter out very small chunks (likely noise)
    min_chunk_size = 100
    filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]
    
    if len(filtered_chunks) < len(chunks):
        logger.info(f"Filtered out {len(chunks) - len(filtered_chunks)} small chunks")
    
    return filtered_chunks
