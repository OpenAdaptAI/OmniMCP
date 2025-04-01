# tests/test_omnimcp_core.py

"""
Tests for core OmniMCP/VisualState functionality using synthetic test images
and a mocked OmniParserClient.
"""

import os
import pytest
import sys

# Make sure patch is imported from unittest.mock
from unittest.mock import patch, MagicMock
from PIL import Image  # Keep needed imports


pytestmark = pytest.mark.skipif(
    sys.platform == "linux" and not os.environ.get("DISPLAY"),
    reason="Requires GUI environment (DISPLAY)",
)

from omnimcp.omnimcp import OmniMCP, VisualState  # noqa: E402
from omnimcp.types import UIElement, ActionVerification  # noqa: E402
from omnimcp.testing_utils import generate_test_ui, generate_action_test_pair  # noqa: E402
from omnimcp.omniparser.client import OmniParserClient  # noqa: E402


# Mock OmniParserClient class for testing VisualState
class MockOmniParserClient:
    """Mock OmniParser client that returns predetermined elements."""

    def __init__(self, elements_to_return: dict):
        self.elements_to_return = elements_to_return
        self.server_url = "http://mock-server:8000"  # Simulate having a server URL

    def parse_image(self, image: Image.Image) -> dict:
        """Mock parse_image method."""
        # Add type hint for clarity
        print("MockOmniParserClient: Returning mock data for parse_image call.")
        return self.elements_to_return

    # Add dummy methods if VisualState or OmniMCP call them during init/update
    def _ensure_server(self):
        pass

    def _check_server(self):
        return True


# Fixture to generate UI data once per module
@pytest.fixture(scope="module")
def synthetic_ui_data():
    # Use the helper function imported from the package
    img, elements_list_of_dicts = generate_test_ui()
    # Create the dict structure the real client's parse_image method returns
    mock_return_data = {"parsed_content_list": elements_list_of_dicts}
    # Return all parts needed by tests
    return img, mock_return_data, elements_list_of_dicts


# Fixture providing an instance of the mock client based on synthetic data
@pytest.fixture
def mock_parser_client(synthetic_ui_data):
    """Fixture providing an instance of MockOmniParserClient."""
    _, mock_parse_return_data, _ = synthetic_ui_data
    return MockOmniParserClient(mock_parse_return_data)


# ----- Tests for VisualState -----


@pytest.mark.asyncio
async def test_visual_state_parsing(synthetic_ui_data, mock_parser_client):
    """Test VisualState.update processes elements from the (mocked) parser client."""
    test_img, _, elements_expected_list_of_dicts = synthetic_ui_data

    # Patch take_screenshot used within visual_state.update
    with patch("omnimcp.omnimcp.take_screenshot", return_value=test_img):
        # Initialize VisualState directly with the mock client instance
        visual_state = VisualState(parser_client=mock_parser_client)
        # Check initial state
        assert not visual_state.elements
        assert visual_state.screen_dimensions is None

        # Call the async update method
        await visual_state.update()

        # Verify state after update
        assert visual_state.screen_dimensions == test_img.size
        assert visual_state._last_screenshot == test_img
        assert visual_state.timestamp is not None

        # Verify elements were processed correctly based on mock data
        # NOTE: The mock data bbox is dict, mapper expects list -> This test WILL FAIL until mock data is fixed!
        # Let's add the bbox fix to generate_test_ui in testing_utils.py first. Assuming that's done:
        assert len(visual_state.elements) == len(elements_expected_list_of_dicts)
        assert all(isinstance(el, UIElement) for el in visual_state.elements)

        # Check a specific element (assuming generate_test_ui puts button first)
        button = next((e for e in visual_state.elements if e.type == "button"), None)
        assert button is not None
        assert button.content == "Submit"
        assert button.id == 0  # Check ID assignment

        # Check element ID assignment is sequential
        assert [el.id for el in visual_state.elements] == list(
            range(len(elements_expected_list_of_dicts))
        )
        print("✅ Visual state parsing test passed (using mock client)")


@pytest.mark.asyncio
async def test_element_finding(synthetic_ui_data, mock_parser_client):
    """Test VisualState.find_element locates elements using basic matching."""
    test_img, _, _ = synthetic_ui_data

    # Patch screenshot and initialize VisualState with mock client
    with patch("omnimcp.omnimcp.take_screenshot", return_value=test_img):
        visual_state = VisualState(parser_client=mock_parser_client)
        await visual_state.update()  # Populate state

        # Test finding known elements (content based on generate_test_ui)
        # Assuming mapping uses list bbox from fixed generate_test_ui and mapping works
        assert len(visual_state.elements) > 0, "Mapping failed, no elements to find"

        button = visual_state.find_element("submit button")
        assert button is not None and button.type == "button"

        textfield = visual_state.find_element(
            "username field"
        )  # Match placeholder/content
        assert textfield is not None and textfield.type == "text_field"

        checkbox = visual_state.find_element("remember checkbox")  # Use type in query
        assert checkbox is not None and checkbox.type == "checkbox"

        link = visual_state.find_element("forgot password")
        assert link is not None and link.type == "link"

        # Test non-existent element
        no_match = visual_state.find_element("non existent pizza")
        assert no_match is None
        print("✅ Element finding test passed (using mock client)")


# ----- Tests for OmniMCP (using mocks) -----


@pytest.mark.asyncio
# Add patches for the controllers used inside OmniMCP.__init__
@patch("omnimcp.omnimcp.OmniParserClient")
@patch("omnimcp.omnimcp.MouseController")
@patch("omnimcp.omnimcp.KeyboardController")
async def test_action_verification(
    mock_kb_controller_class,  # Order matters, matches decorators bottom-up
    mock_mouse_controller_class,
    mock_omniparser_client_class,
    # synthetic_ui_data # Fixture not actually used in this specific test logic
):
    """Test the basic pixel diff action verification in OmniMCP."""
    # Mock the client instance
    mock_client_instance = MagicMock(spec=OmniParserClient)
    mock_client_instance.server_url = "http://mock-server:8000"
    mock_client_instance.parse_image.return_value = {"parsed_content_list": []}
    mock_omniparser_client_class.return_value = mock_client_instance

    # Mock the controller instances (optional, patching class often enough)
    # mock_mouse_controller_class.return_value = MagicMock(spec=MouseController)
    # mock_kb_controller_class.return_value = MagicMock(spec=KeyboardController)

    # Generate before/after images
    before_click, after_click, _ = generate_action_test_pair("click", "button")
    before_type, after_type, _ = generate_action_test_pair("type", "text_field")
    before_check, after_check, _ = generate_action_test_pair("check", "checkbox")
    no_change_img, _, _ = generate_action_test_pair("click", "link")

    # Create OmniMCP instance - its internal controller creation will now use mocks
    mcp = OmniMCP()
    # Manually set screen dimensions if needed by _verify_action
    mcp._visual_state.screen_dimensions = before_click.size

    # --- Test verification logic ---
    click_verification = await mcp._verify_action(before_click, after_click)
    assert isinstance(click_verification, ActionVerification)
    assert click_verification.success is True, "Click action verification failed"
    assert click_verification.confidence > 0.01

    type_verification = await mcp._verify_action(before_type, after_type)
    assert isinstance(type_verification, ActionVerification)
    assert type_verification.success is True, "Type action verification failed"
    assert type_verification.confidence > 0.01

    check_verification = await mcp._verify_action(before_check, after_check)
    assert isinstance(check_verification, ActionVerification)
    assert check_verification.success is True, "Check action verification failed"
    assert check_verification.confidence > 0.01

    no_change_verification = await mcp._verify_action(no_change_img, no_change_img)
    assert isinstance(no_change_verification, ActionVerification)
    assert no_change_verification.success is False, (
        "No change action verification failed"
    )
    assert no_change_verification.confidence == 0.0
    print("✅ Action verification test passed")
