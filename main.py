"""Main pipeline orchestrator for PDF to instruction dataset conversion."""

import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DATASET_FILE, RAW_PDF_DIR
from src.generator import test_ollama_connection
from src.pipeline import append_dataset_entries, ensure_processed_dir, process_pdf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("PDF to Instruction Dataset Pipeline")
    logger.info("=" * 60)

    # Test Ollama connection
    if not test_ollama_connection():
        logger.error("Cannot proceed without Ollama connection")
        return

    # Setup directories
    raw_dir = Path(RAW_PDF_DIR)
    ensure_processed_dir()

    # Find all PDF files
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}")
        logger.info(f"Please place your PDF files in: {raw_dir.absolute()}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s)")

    # Process all PDFs
    all_entries = []
    for pdf_path in pdf_files:
        entries = process_pdf(pdf_path)
        all_entries.extend(entries)
    # Append to JSONL (append to preserve prior runs)
    output_path = Path(OUTPUT_DATASET_FILE)
    logger.info(f"Appending {len(all_entries)} entries to {output_path}")
    append_dataset_entries(all_entries, output_path)

    logger.info("=" * 60)
    logger.info(f"✓ Pipeline complete!")
    logger.info(f"✓ Generated {len(all_entries)} instruction-response pairs")
    logger.info(f"✓ Output saved to: {output_path.absolute()}")
    logger.info("=" * 60)

    # Show sample entry
    if all_entries:
        logger.info("\nSample entry:")
        print(json.dumps(all_entries[0], indent=2))


if __name__ == "__main__":
    main()
