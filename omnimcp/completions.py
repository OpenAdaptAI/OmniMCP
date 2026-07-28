# omnimcp/completions.py

import json
import time
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import config  # Import config for API key and model name
from .utils import logger  # Reuse logger from utils

# Type variable for the Pydantic response model
T = TypeVar("T", bound=BaseModel)

# --- Client Initialization ---
# Initialize based on configured provider (currently only Anthropic)
# TODO: Add support for other providers (OpenAI, Google) based on config.LLM_PROVIDER
if config.LLM_PROVIDER.lower() == "anthropic":
    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment/config for Anthropic provider."
        )
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize Anthropic client: {e}")
        raise
else:
    # In the future, add client init for other providers here
    logger.warning(
        f"LLM Provider '{config.LLM_PROVIDER}' not yet fully supported in completions.py. Falling back/failing."
    )
    # For now, raise error if not anthropic
    raise NotImplementedError(
        f"LLM provider '{config.LLM_PROVIDER}' integration not implemented."
    )


# --- Retry Configuration ---
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    # Add other provider-specific retryable errors here if needed
)
MAX_RETRIES = 3


# --- Helper to format messages for logging ---
def format_chat_messages(messages: list[dict[str, str]]) -> str:
    """Format chat messages in a readable way for logs."""
    result = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        result.append("=" * 40 + f" {role} " + "=" * 40)
        result.append(content)
    result.append("=" * 80 + "=" * (len(" ASSISTANT ") // 2))  # End marker
    return "\n".join(result)


# --- Core API Call Function ---
@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_random_exponential(min=1, max=30),  # Exponential backoff up to 30s
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=lambda retry_state: logger.warning(
        f"LLM API Error (Attempt {retry_state.attempt_number}/{MAX_RETRIES}): "
        f"{retry_state.outcome.exception()}. Retrying...",
    ),
    reraise=True,  # Reraise the exception after retries are exhausted
)
def call_llm_api(
    messages: list[dict[str, str]],
    response_model: type[T],
    model: str | None = None,  # Allow overriding config default
    temperature: float = 0.1,  # Lower temperature for more deterministic planning
    system_prompt: str | None = None,
) -> T:
    """
    Calls the configured LLM API, expecting a JSON response conforming to the pydantic model.

    Args:
        messages: List of message dictionaries (e.g., [{"role": "user", "content": ...}]).
        response_model: The Pydantic model class for the expected JSON structure.
        model: Optional override for the LLM model name.
        temperature: The sampling temperature.
        system_prompt: Optional system prompt string.

    Returns:
        An instance of the response_model Pydantic model.

    Raises:
        anthropic.APIError: If a non-retryable Anthropic API error occurs.
        ValueError: If the response is not valid JSON or doesn't match the schema.
        NotImplementedError: If the configured LLM provider isn't supported.
        RetryError: If the call fails after all retry attempts.
        Exception: For other unexpected errors.
    """

    if config.DEBUG_FULL_PROMPTS:
        formatted_messages = format_chat_messages(messages)
        logger.debug(f"Formatted messages being sent:\n{formatted_messages}")

    start_time = time.time()

    # --- API Specific Call Logic ---
    # TODO: Add conditional logic here for different providers based on config.LLM_PROVIDER
    if config.LLM_PROVIDER.lower() == "anthropic":
        # Use provided model or default from config
        model_to_use = model or config.ANTHROPIC_DEFAULT_MODEL
        logger.debug(
            f"Calling LLM API (model: {model_to_use}) with {len(messages)} messages."
        )
        try:
            api_response = client.messages.create(
                model=model_to_use,
                messages=messages,
                system=system_prompt,
                max_tokens=2048,  # Adjust needed token count
                temperature=temperature,
            )
            # Extract text response - specific to Anthropic's Messages API format
            if (
                not api_response.content
                or not isinstance(api_response.content, list)
                or not hasattr(api_response.content[0], "text")
            ):
                logger.error(
                    f"Unexpected Anthropic API response structure: {api_response}"
                )
                raise ValueError(
                    "Could not extract text content from Anthropic response."
                )
            response_text = api_response.content[0].text.strip()

        except anthropic.APIError as e:  # Catch specific non-retryable Anthropic errors
            logger.error(f"Non-retryable Anthropic API error: {type(e).__name__} - {e}")
            raise  # Reraise non-retryable or errors hitting max retries
        except Exception as e:  # Catch other unexpected errors during API call
            logger.error(
                f"Unexpected error calling Anthropic API: {type(e).__name__} - {e}",
                exc_info=True,
            )
            raise
    else:
        # Should have been caught by client init, but safeguard here
        raise NotImplementedError(
            f"API call logic for provider '{config.LLM_PROVIDER}' not implemented."
        )
    # --- End API Specific Call Logic ---

    duration_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"LLM API call completed in {duration_ms}ms.")
    logger.debug(f"Raw LLM response text:\n{response_text}")

    # Clean potential markdown code fences (common issue)
    response_text = response_text.removeprefix("```json")
    response_text = response_text.removesuffix("```")
    response_text = response_text.strip()

    # Parse and validate the JSON response using the Pydantic model
    try:
        parsed_response = response_model.model_validate_json(response_text)
        logger.info(f"Successfully parsed LLM response into {response_model.__name__}.")
        return parsed_response
    except ValidationError as e:
        logger.error(
            f"Failed to validate LLM JSON response against schema {response_model.__name__}."
        )
        logger.error(f"Validation Errors: {e}")
        logger.error(f"Response JSON text was: {response_text}")
        # Don't raise e directly, wrap it
        raise ValueError(f"LLM response did not match the expected format: {e}") from e
    except json.JSONDecodeError as e:
        logger.error("Failed to decode LLM response as JSON.")
        logger.error(f"Raw response text was: {response_text}")
        raise ValueError(f"LLM response was not valid JSON: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during Pydantic validation: {e}", exc_info=True)
        raise  # Reraise unexpected validation errors
