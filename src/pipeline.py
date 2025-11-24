"""Shared pipeline utilities for processing PDFs into instruction data."""

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jsonlines

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OUTPUT_DATASET_FILE,
    PROCESSED_DATA_DIR,
)
from src.chunker import chunk_text
from src.extractor import clean_text, extract_text_from_pdf
from src.generator import generate_instruction

logger = logging.getLogger(__name__)

# Serialize concurrent writes to the dataset file so multiple background tasks
# cannot interleave their JSONL entries.
DATASET_WRITE_LOCK = threading.Lock()


def ensure_processed_dir() -> Path:
    """Guarantee the processed data directory exists."""
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def append_dataset_entries(
    entries: List[dict], output_path: Optional[Path] = None
) -> None:
    """
    Append a list of entries to the dataset JSONL file.

    Args:
        entries: Instruction data to append
        output_path: Optional override path for the dataset file
    """
    if not entries:
        return

    ensure_processed_dir()
    dataset_path = Path(output_path) if output_path else Path(OUTPUT_DATASET_FILE)

    with DATASET_WRITE_LOCK:
        with jsonlines.open(dataset_path, mode="a") as writer:
            writer.write_all(entries)


def process_pdf(
    pdf_path: Path,
    model: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> List[dict]:
    """
    Process a single PDF file and return generated instruction entries.

    Args:
        pdf_path: Path to the PDF file
        model: Optional model name to use for generation
        device: Optional device hint (e.g., "cpu" to force CPU)
        progress_callback: Optional callable to report progress events

    Returns:
        List of instruction dataset entries
    """

    def report(stage: str, **kwargs) -> None:
        if progress_callback:
            progress_callback({"stage": stage, **kwargs})

    logger.info("Processing PDF: %s", pdf_path.name)
    report("starting", file=pdf_path.name)

    raw_text = extract_text_from_pdf(str(pdf_path))
    if not raw_text:
        report("failed", file=pdf_path.name, reason="extraction_failed")
        logger.error("Failed to extract text from %s", pdf_path.name)
        return []

    cleaned_text = clean_text(raw_text)
    report("cleaned", file=pdf_path.name, characters=len(cleaned_text))
    logger.info("Cleaned text: %s characters", len(cleaned_text))

    chunks = chunk_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
    total_chunks = len(chunks)
    report("chunked", file=pdf_path.name, total_chunks=total_chunks)
    logger.info("Created %s chunks", total_chunks)

    if not total_chunks:
        report("failed", file=pdf_path.name, reason="no_chunks")
        logger.error("No chunks produced for %s", pdf_path.name)
        return []

    dataset_entries = []
    for i, chunk in enumerate(chunks):
        report(
            "generating",
            file=pdf_path.name,
            chunk_index=i,
            total_chunks=total_chunks,
            progress=(i / total_chunks),
        )
        instruction_data = generate_instruction(chunk, model=model, device=device)
        if instruction_data:
            instruction_data["source_file"] = pdf_path.name
            instruction_data["chunk_index"] = i
            dataset_entries.append(instruction_data)
        else:
            logger.warning("Failed to generate instruction for chunk %s", i)

    report(
        "completed",
        file=pdf_path.name,
        total_chunks=total_chunks,
        generated=len(dataset_entries),
    )
    logger.info(
        "Generated %s instruction entries from %s", len(dataset_entries), pdf_path.name
    )
    return dataset_entries
