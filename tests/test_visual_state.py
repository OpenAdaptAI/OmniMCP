# tests/test_omnimcp_core.py

from unittest.mock import patch

import pytest
from PIL import Image

# Removed: from omnimcp.mcp_server import OmniMCP (no longer used in this file)
from omnimcp.synthetic_ui import generate_login_screen

# Corrected imports based on file moves
from omnimcp.visual_state import VisualState

# --- Fixtures ---


# Mock OmniParserClient for testing VisualState without real API calls
class MockOmniParserClient:
    def __init__(self, mock_response: dict):
        self.mock_response = mock_response
        self.server_url = "http://mock-parser.test"

    def parse_image(self, image: Image.Image) -> dict:
        # Simulate returning the mock response regardless of image input
        return self.mock_response


@pytest.fixture
def synthetic_ui_data():
    """
    Generates synthetic UI data and formats it like the expected parser response.
    Returns: (PIL.Image, dict_parser_response, list_of_expected_parser_dicts)
    """
    img, elements_obj_list = generate_login_screen()

    # Convert UIElement objects to dicts matching expected OmniParser JSON structure
    mock_parser_list = []
    for el in elements_obj_list:
        x, y, w, h = el.bounds
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        # Create dict matching expected parser structure key "bbox" and list format
        parser_dict = {
            "bbox": [x_min, y_min, x_max, y_max],  # Use "bbox" key with list
            "content": el.content,
            "type": el.type,
            "confidence": el.confidence,  # Ensure confidence is included
            "attributes": el.attributes,
        }
        mock_parser_list.append(parser_dict)

    # The final structure expected by VisualState._update_elements_from_parser
    mock_parser_response = {"parsed_content_list": mock_parser_list}

    return img, mock_parser_response, mock_parser_list


@pytest.fixture
def mock_parser_client(synthetic_ui_data):
    """Provides a MockOmniParserClient instance with synthetic data."""
    _, mock_parser_response, _ = synthetic_ui_data
    return MockOmniParserClient(mock_parser_response)


# --- Tests ---

# TODO: Add test for VisualState initialization failure if parser client is None/invalid.
# def test_visual_state_initialization_fails_without_client():
#     with pytest.raises(ValueError):
#         VisualState(parser_client=None)


def test_visual_state_parsing(synthetic_ui_data, mock_parser_client):
    """Test VisualState.update processes elements from the (mocked) parser client."""
    test_img, _, expected_elements_list_of_dicts = synthetic_ui_data

    # Patch take_screenshot used within visual_state.update
    # Target the function where it's looked up (in the visual_state module)
    with patch("omnimcp.visual_state.take_screenshot", return_value=test_img):
        vs = VisualState(parser_client=mock_parser_client)
        vs.update()  # Trigger screenshot mock and parsing mock

    # Assertions
    assert vs._last_screenshot == test_img
    assert vs.screen_dimensions == test_img.size
    assert len(vs.elements) == len(expected_elements_list_of_dicts)

    # Compare element details (convert actual UIElements back to dicts for comparison)
    actual_elements_dicts = [elem.to_dict() for elem in vs.elements]

    # Basic check: Ensure IDs are sequential starting from 0
    assert all(
        actual_elements_dicts[i]["id"] == i for i in range(len(actual_elements_dicts))
    )

    # Compare content based on expected list (ignoring ID for comparison)
    expected_contents = {
        (d["type"], d["content"]) for d in expected_elements_list_of_dicts
    }
    actual_contents = {(d["type"], d["content"]) for d in actual_elements_dicts}
    assert actual_contents == expected_contents


def test_element_finding(synthetic_ui_data, mock_parser_client):
    """Test VisualState.find_element locates elements using basic matching."""
    test_img, _, _ = synthetic_ui_data

    # Patch take_screenshot used within visual_state.update
    with patch("omnimcp.visual_state.take_screenshot", return_value=test_img):
        vs = VisualState(parser_client=mock_parser_client)
        vs.update()  # Populate elements

    # TODO: Improve find_element logic and add more robust tests here.
    # Current matching is very basic keyword search.

    # Test finding existing elements
    login_button = vs.find_element("Login button")
    assert login_button is not None
    assert login_button.type == "button"
    assert login_button.content == "Login"

    username_field = vs.find_element("username text field")
    assert username_field is not None
    assert username_field.type == "text_field"
    # Expect empty string for initial content of the username field
    assert username_field.content == ""

    # Test finding non-existent element
    non_existent = vs.find_element("non_existent element foobar")
    assert non_existent is None

    # Test finding based only on type (might be ambiguous)
    a_button = vs.find_element("button")
    assert a_button is not None  # Should find *a* button
