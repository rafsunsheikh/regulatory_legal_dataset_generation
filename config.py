"""Configuration settings for the PDF to instruction dataset pipeline."""

# Ollama Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama endpoint

# Chunking Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

# Paths
RAW_PDF_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DATASET_FILE = "data/processed/dataset.jsonl"

# Generation Configuration
TEMPERATURE = 0.7  # Controls randomness (0.0 = deterministic, 1.0 = creative)
MAX_RETRIES = 3  # Number of retries for failed API calls

# Prompt Template for Instruction Generation
INSTRUCTION_PROMPT = """You are an expert in legal regulations. Based on the following legal text, generate a high-quality instruction-response pair for fine-tuning a language model.

Legal Text:
{chunk}

Generate a JSON object with the following structure:
{{
  "instruction": "A clear question or task related to this legal text",
  "input": "Any additional context if needed (can be empty)",
  "output": "A comprehensive answer based on the legal text"
}}

Important:
- The instruction should be specific and answerable from the text
- The output should be accurate and cite relevant parts of the regulation
- Keep the instruction natural and varied (don't always start with "What is...")

Generate only the JSON object, no additional text."""
