# test_deploy_and_parse.py
"""
A simple script to test OmniParser deployment and basic image parsing.
Reuses config loading from omnimcp.config.
"""

import sys
import json
from PIL import Image

# Import config first to trigger .env loading
from omnimcp.config import config
from omnimcp.utils import logger, take_screenshot
from omnimcp.omniparser.client import OmniParserClient


if __name__ == "__main__":
    logger.info("--- Starting OmniParser Deployment and Parse Test ---")

    # Optional: Check if config loaded AWS keys (for user feedback)
    # Note: boto3 might still find credentials via ~/.aws/credentials even if not in .env/env vars
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY and config.AWS_REGION:
        logger.info(
            f"AWS config loaded via pydantic-settings (Region: {config.AWS_REGION})."
        )
    else:
        logger.warning(
            "AWS credentials/region not found via config (env vars or .env)."
        )
        logger.warning(
            "Ensure credentials are configured where boto3 can find them (e.g., ~/.aws/credentials, env vars)."
        )

    # 1. Initialize Client (Triggers auto-deploy/discovery)
    logger.info(
        "Initializing OmniParserClient (this may take several minutes if deploying)..."
    )
    try:
        parser_client = OmniParserClient(
            auto_deploy=True
        )  # auto_deploy=True is default
        logger.success(
            f"OmniParserClient ready. Connected to server: {parser_client.server_url}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize OmniParserClient: {e}", exc_info=True)
        logger.error(
            "Please check AWS credentials configuration and network connectivity."
        )
        sys.exit(1)

    # 2. Take Screenshot
    logger.info("Taking screenshot...")
    try:
        screenshot: Image.Image = take_screenshot()
        logger.success("Screenshot taken successfully.")
        try:
            screenshot_path = "test_deploy_screenshot.png"
            screenshot.save(screenshot_path)
            logger.info(f"Saved screenshot for debugging to: {screenshot_path}")
        except Exception as save_e:
            logger.warning(f"Could not save debug screenshot: {save_e}")
    except Exception as e:
        logger.error(f"Failed to take screenshot: {e}", exc_info=True)
        sys.exit(1)

    # 3. Parse Image
    logger.info(f"Sending screenshot to OmniParser at {parser_client.server_url}...")
    results = None
    try:
        results = parser_client.parse_image(screenshot)
        logger.success("Received response from OmniParser.")
    except Exception as e:
        logger.error(
            f"Unexpected error during client.parse_image call: {e}", exc_info=True
        )
        sys.exit(1)

    # 4. Print Results
    if isinstance(results, dict) and "error" in results:
        logger.error(f"OmniParser server returned an error: {results['error']}")
    elif isinstance(results, dict):
        logger.success("OmniParser returned a successful response.")
        logger.info("Raw JSON Result:")
        try:
            print(json.dumps(results, indent=2))
        except Exception as json_e:
            logger.error(f"Could not format result as JSON: {json_e}")
            print(results)
    else:
        logger.warning(
            f"Received unexpected result format from OmniParser client: {type(results)}"
        )
        print(results)

    logger.info("--- Test Finished ---")
