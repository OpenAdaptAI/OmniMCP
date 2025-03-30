# tests/test_synthetic_ui.py
import pytest
from PIL import Image
from unittest.mock import MagicMock  # Simple way to create dummy plan object

# Assuming your package structure allows this import
from omnimcp.synthetic_ui import (
    generate_login_screen,
    generate_logged_in_screen,
    simulate_action,
    draw_highlight,  # Test a utility if desired
)
from omnimcp.types import UIElement


# --- Fixtures ---


@pytest.fixture
def login_state() -> tuple[Image.Image, list[UIElement]]:
    """Provides the initial login screen state."""
    img, elements = generate_login_screen()
    return img, elements


@pytest.fixture
def logged_in_state() -> tuple[Image.Image, list[UIElement]]:
    """Provides the logged-in screen state."""
    img, elements = generate_logged_in_screen(username="testuser")
    return img, elements


# --- Tests for Generation ---


def test_generate_login_screen(login_state):
    """Test login screen generation basics."""
    img, elements = login_state
    assert isinstance(img, Image.Image)
    assert isinstance(elements, list)
    assert len(elements) == 5  # Assuming 5 interactive elements generated
    assert all(isinstance(el, UIElement) for el in elements)
    # Check if login button is present (assuming ID 4 based on generation logic)
    login_button = next((el for el in elements if el.id == 4), None)
    assert login_button is not None
    assert login_button.type == "button"
    assert login_button.content == "Login"


def test_generate_logged_in_screen(logged_in_state):
    """Test logged-in screen generation basics."""
    img, elements = logged_in_state
    assert isinstance(img, Image.Image)
    assert isinstance(elements, list)
    assert len(elements) > 0  # Should have at least welcome text and logout
    assert elements[0].type == "text"  # Welcome message
    assert "testuser" in elements[0].content


# --- Tests for Simulation ---


def test_simulate_action_type_username(login_state):
    """Test simulating typing into the username field."""
    img, elements = login_state
    # Create a mock plan object with necessary attributes
    plan = MagicMock()
    plan.action = "type"
    plan.element_id = 0  # Username field ID
    plan.text_to_type = "testuser"

    new_img, new_elements = simulate_action(img, elements, plan)

    assert elements[0].content == ""  # Original should be unchanged
    assert new_elements[0].content == "testuser"
    assert new_elements[1].content == ""  # Password field unchanged
    assert id(new_img) != id(img)  # Image object should have been modified (copied)
    assert new_elements is not elements  # List should be a deep copy


def test_simulate_action_type_password(login_state):
    """Test simulating typing into the password field."""
    img, elements = login_state
    plan = MagicMock()
    plan.action = "type"
    plan.element_id = 1  # Password field ID
    plan.text_to_type = "password123"

    new_img, new_elements = simulate_action(img, elements, plan)

    assert new_elements[1].content == "password123"  # Check internal content
    # We don't easily check the visual masking ('***') here, focus on state change
    assert new_elements[0].content == ""  # Username field unchanged


def test_simulate_action_click_checkbox_toggle(login_state):
    """Test simulating clicking the checkbox toggles its state."""
    img, elements = login_state
    plan = MagicMock()
    plan.action = "click"
    plan.element_id = 2  # Checkbox ID

    # First click (check)
    img_after_check, elements_after_check = simulate_action(img, elements, plan)
    assert elements_after_check[2].attributes["checked"] is True
    assert elements[2].attributes["checked"] is False  # Original unchanged

    # Second click (uncheck)
    img_after_uncheck, elements_after_uncheck = simulate_action(
        img_after_check, elements_after_check, plan
    )
    assert elements_after_uncheck[2].attributes["checked"] is False


def test_simulate_action_click_login_success(login_state):
    """Test simulating clicking login when fields are filled."""
    img, elements = login_state
    # Pre-fill the elements list state for the test
    elements[0].content = "testuser"
    elements[1].content = "password123"

    plan = MagicMock()
    plan.action = "click"
    plan.element_id = 4  # Login button ID

    new_img, new_elements = simulate_action(
        img, elements, plan, username_for_login="testuser"
    )

    # Expect state transition to logged-in screen
    assert len(new_elements) < len(elements)  # Logged in screen has fewer elements
    assert new_elements[0].type == "text"
    assert "Welcome, testuser!" in new_elements[0].content


def test_simulate_action_click_login_fail(login_state):
    """Test simulating clicking login when fields are empty."""
    img, elements = login_state
    plan = MagicMock()
    plan.action = "click"
    plan.element_id = 4  # Login button ID

    new_img, new_elements = simulate_action(img, elements, plan)

    # Expect no state transition
    assert len(new_elements) == len(elements)
    assert new_elements[0].content == ""  # Username still empty
    # Could also check image identity, but copy might happen anyway
    # assert id(new_img) == id(img)


# --- Test for Visualization (Basic) ---


def test_draw_highlight(login_state):
    """Test that draw_highlight runs and returns an image."""
    img, elements = login_state
    element_to_highlight = elements[0]  # Highlight username field
    plan = MagicMock()  # Dummy plan for the function signature
    plan.action = "type"
    plan.text_to_type = "dummy"

    highlighted_img = draw_highlight(img, element_to_highlight, plan=plan)

    assert isinstance(highlighted_img, Image.Image)
    assert highlighted_img.size == img.size
