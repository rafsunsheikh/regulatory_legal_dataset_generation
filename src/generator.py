"""Instruction generation module using Ollama."""

import json
import logging
import time
from typing import Dict, Optional

import ollama

from config import INSTRUCTION_PROMPT, MAX_RETRIES, OLLAMA_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_instruction(
    chunk: str,
    model: Optional[str] = None,
    device: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    prompt: Optional[str] = None,
    retry_count: int = 0,
) -> Optional[Dict[str, str]]:
    """
    Generate an instruction-response pair from a text chunk using Ollama.

    Args:
        chunk: Text chunk to generate instruction from
        model: Optional model name to use (defaults to config OLLAMA_MODEL)
        retry_count: Current retry attempt

    Returns:
        Dictionary with 'instruction', 'input', and 'output' keys, or None if failed
    """
    if not chunk or not chunk.strip():
        logger.warning("Empty chunk provided for instruction generation")
        return None

    try:
        # Format the prompt with the chunk
        prompt_text = (prompt or INSTRUCTION_PROMPT).format(chunk=chunk)
        target_model = model or OLLAMA_MODEL
        target_temperature = TEMPERATURE if temperature is None else temperature
        retry_limit = MAX_RETRIES if max_retries is None else max_retries

        # Call Ollama API
        logger.info(f"Generating instruction with {target_model}...")
        options = {
            "temperature": target_temperature,
        }
        # If explicitly CPU, force no GPU layers.
        if device == "cpu":
            options["num_gpu"] = 0

        response = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": prompt_text}],
            options=options,
        )

        # Extract the response content
        content = response["message"]["content"].strip()

        # Try to parse JSON from the response
        # Sometimes the model wraps JSON in markdown code blocks
        if content.startswith("```"):
            # Remove markdown code block markers
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Parse JSON
        result = json.loads(content)

        # Validate the structure
        required_keys = {"instruction", "input", "output"}
        if not all(key in result for key in required_keys):
            logger.error(f"Missing required keys in response: {result.keys()}")
            raise ValueError("Invalid response structure")

        logger.info("Successfully generated instruction")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response content: {content}")

        if retry_count < retry_limit:
            logger.info(f"Retrying... (attempt {retry_count + 1}/{retry_limit})")
            time.sleep(2)  # Brief delay before retry
            return generate_instruction(
                chunk,
                model=model,
                device=device,
                prompt=prompt,
                temperature=temperature,
                max_retries=max_retries,
                retry_count=retry_count + 1,
            )
        return None

    except Exception as e:
        logger.error(f"Error generating instruction: {e}")

        if retry_count < retry_limit:
            logger.info(f"Retrying... (attempt {retry_count + 1}/{retry_limit})")
            time.sleep(2)
            return generate_instruction(
                chunk,
                model=model,
                device=device,
                prompt=prompt,
                temperature=temperature,
                max_retries=max_retries,
                retry_count=retry_count + 1,
            )
        return None


def test_ollama_connection(model: Optional[str] = None) -> bool:
    """
    Test if Ollama is running and the model is available.

    Returns:
        True if connection successful, False otherwise
    """
    target_model = model or OLLAMA_MODEL
    try:
        logger.info(f"Testing connection to Ollama with model {target_model}...")
        response = ollama.chat(
            model=target_model, messages=[{"role": "user", "content": "Hello"}]
        )
        logger.info("✓ Ollama connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to Ollama: {e}")
        logger.error(
            f"Make sure Ollama is running and model '{target_model}' is installed"
        )
