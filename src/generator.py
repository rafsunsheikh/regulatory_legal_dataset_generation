"""Instruction generation module using Ollama."""

import json
import logging
import time
from typing import Dict, Optional
import ollama
from config import (
    OLLAMA_MODEL,
    INSTRUCTION_PROMPT,
    TEMPERATURE,
    MAX_RETRIES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_instruction(chunk: str, retry_count: int = 0) -> Optional[Dict[str, str]]:
    """
    Generate an instruction-response pair from a text chunk using Ollama.
    
    Args:
        chunk: Text chunk to generate instruction from
        retry_count: Current retry attempt
        
    Returns:
        Dictionary with 'instruction', 'input', and 'output' keys, or None if failed
    """
    if not chunk or not chunk.strip():
        logger.warning("Empty chunk provided for instruction generation")
        return None
    
    try:
        # Format the prompt with the chunk
        prompt = INSTRUCTION_PROMPT.format(chunk=chunk)
        
        # Call Ollama API
        logger.info(f"Generating instruction with {OLLAMA_MODEL}...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': TEMPERATURE,
            }
        )
        
        # Extract the response content
        content = response['message']['content'].strip()
        
        # Try to parse JSON from the response
        # Sometimes the model wraps JSON in markdown code blocks
        if content.startswith('```'):
            # Remove markdown code block markers
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()
        
        # Parse JSON
        result = json.loads(content)
        
        # Validate the structure
        required_keys = {'instruction', 'input', 'output'}
        if not all(key in result for key in required_keys):
            logger.error(f"Missing required keys in response: {result.keys()}")
            raise ValueError("Invalid response structure")
        
        logger.info("Successfully generated instruction")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response content: {content}")
        
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying... (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(2)  # Brief delay before retry
            return generate_instruction(chunk, retry_count + 1)
        return None
        
    except Exception as e:
        logger.error(f"Error generating instruction: {e}")
        
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying... (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(2)
            return generate_instruction(chunk, retry_count + 1)
        return None


def test_ollama_connection() -> bool:
    """
    Test if Ollama is running and the model is available.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        logger.info(f"Testing connection to Ollama with model {OLLAMA_MODEL}...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        logger.info("✓ Ollama connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to Ollama: {e}")
        logger.error(f"Make sure Ollama is running and model '{OLLAMA_MODEL}' is installed")
        logger.error(f"Install with: ollama pull {OLLAMA_MODEL}")
        return False
