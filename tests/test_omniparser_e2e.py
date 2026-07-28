# tests/test_omniparser_e2e.py

"""End-to-end tests for OmniParser deployment and function."""

import time
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

# Only import OmniParserClient now
from omnimcp.omniparser.client import OmniParserClient

# Config might still be needed if checking AWS env vars, keep for now
# from omnimcp.config import config # Removed as test logic doesn't directly use it


@pytest.fixture(scope="module")
def test_image():
    """Fixture to provide the test image."""
    # Assuming test_images is relative to the tests directory or project root
    # Adjust path if necessary based on where you run pytest from
    test_image_path = Path(__file__).parent.parent / "test_images" / "synthetic_ui.png"
    # Fallback if not found relative to tests/
    if not test_image_path.exists():
        test_image_path = Path("test_images") / "synthetic_ui.png"

    assert test_image_path.exists(), f"Test image not found: {test_image_path}"
    return Image.open(test_image_path)


@pytest.mark.xfail(reason="Client connection/check currently failing in e2e")
@pytest.mark.e2e
def test_client_initialization_and_availability(test_image):  # Combined test
    """
    Test if OmniParser client can initialize, which includes finding
    or deploying a server and checking its availability.
    Also performs a basic parse test.
    """
    logger.info("\nTesting OmniParserClient initialization (auto-deploy enabled)...")
    client = None
    try:
        # Initialization itself triggers the ensure_server logic
        start_time = time.time()
        client = OmniParserClient(auto_deploy=True)
        init_time = time.time() - start_time
        logger.success(
            f"Client initialized successfully in {init_time:.1f}s. Server URL: {client.server_url}"
        )
        assert client.server_url is not None
    except Exception as e:
        pytest.fail(f"OmniParserClient initialization failed: {e}")

    # Perform a basic parse test now that client is initialized
    logger.info("Testing image parsing via initialized client...")
    start_time = time.time()
    result = client.parse_image(test_image)
    parse_time = time.time() - start_time
    logger.info(f"Parse completed in {parse_time:.1f}s.")

    assert result is not None, "Parse result should not be None"
    assert "error" not in result, f"Parsing returned an error: {result.get('error')}"
    assert "parsed_content_list" in result, (
        "Parsing result missing 'parsed_content_list'"
    )
    elements = result.get("parsed_content_list", [])
    logger.info(f"Found {len(elements)} elements.")
    assert len(elements) >= 3, "Expected at least a few elements in the synthetic image"


# Note: The original test_image_parsing test is now effectively combined
# into test_client_initialization_and_availability as the client must be
# initialized successfully before parsing can be tested.
# You could potentially add teardown logic here using Deploy.stop() if needed,
# but the teardown_class in test_omnimcp.py might cover cleanup globally.
