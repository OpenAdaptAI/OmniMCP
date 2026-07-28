# tests/test_core.py
import pytest

# Assuming imports work based on installation/path
from omnimcp.core import LLMActionPlan, plan_action_for_ui
from omnimcp.types import Bounds, UIElement

# --- Fixture for Sample Elements ---


@pytest.fixture
def sample_elements() -> list[UIElement]:
    """Provides a sample list of UIElements similar to the login screen."""
    # Simplified bounds for brevity
    bounds: Bounds = (0.1, 0.1, 0.2, 0.05)
    return [
        UIElement(
            id=0,
            type="text_field",
            content="",
            bounds=bounds,
            attributes={"label": "Username:"},
        ),
        UIElement(
            id=1,
            type="text_field",
            content="",
            bounds=bounds,
            attributes={"is_password": True, "label": "Password:"},
        ),
        UIElement(
            id=2,
            type="checkbox",
            content="Remember Me",
            bounds=bounds,
            attributes={"checked": False},
        ),
        UIElement(id=3, type="link", content="Forgot Password?", bounds=bounds),
        UIElement(id=4, type="button", content="Login", bounds=bounds),
    ]


# --- Tests for plan_action_for_ui ---


# Use pytest-mock's 'mocker' fixture
def test_plan_action_step1_type_user(mocker, sample_elements):
    """Test planning the first step: typing username."""
    user_goal = "Log in as testuser with password pass"
    action_history = []

    # Mock the LLM API call within the core module
    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # Configure the mock to return a specific plan
    mock_plan_step1 = LLMActionPlan(
        reasoning="Need to type username first.",
        action="type",
        element_id=0,
        text_to_type="testuser",
        is_goal_complete=False,
    )
    mock_llm_api.return_value = mock_plan_step1

    # Call the function under test
    llm_plan_result, target_element_result = plan_action_for_ui(
        elements=sample_elements, user_goal=user_goal, action_history=action_history
    )

    # Assertions
    mock_llm_api.assert_called_once()  # Check API was called
    call_args, _call_kwargs = mock_llm_api.call_args
    # Check prompt content (basic check)
    messages = call_args[0]
    assert user_goal in messages[0]["content"]
    assert (
        sample_elements[0].to_prompt_repr() in messages[0]["content"]
    )  # Check element rendering
    # assert "Previous Actions Taken:\n- None" in messages[0]['content'] # Check history rendering
    # Check prompt content (basic check)
    messages = call_args[0]
    prompt_text = messages[0]["content"]  # Get the rendered prompt text
    assert user_goal in prompt_text
    assert sample_elements[0].to_prompt_repr() in prompt_text  # Check element rendering
    # Check history rendering more robustly
    assert "**Previous Actions Taken:**" in prompt_text
    assert "- None" in prompt_text  # Check that '- None' appears when history is empty
    # Check returned values
    assert llm_plan_result == mock_plan_step1
    assert target_element_result is not None
    assert target_element_result.id == 0


def test_plan_action_step3_click_login(mocker, sample_elements):
    """Test planning the third step: clicking login and completing goal."""
    user_goal = "Log in as testuser with password pass"
    # Simulate state where fields are filled
    sample_elements[0].content = "testuser"
    sample_elements[1].content = "pass"  # Content updated internally
    action_history = ["Action: type 'testuser'...", "Action: type 'pass'..."]

    # Mock the LLM API call
    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # Configure mock for step 3 response
    mock_plan_step3 = LLMActionPlan(
        reasoning="Fields filled, clicking Login.",
        action="click",
        element_id=4,
        text_to_type=None,
        is_goal_complete=True,  # Goal completes on this step
    )
    mock_llm_api.return_value = mock_plan_step3

    # Call the function
    llm_plan_result, target_element_result = plan_action_for_ui(
        elements=sample_elements, user_goal=user_goal, action_history=action_history
    )

    # Assertions
    mock_llm_api.assert_called_once()
    call_args, _call_kwargs = mock_llm_api.call_args
    messages = call_args[0]
    # Check history rendering in prompt
    assert action_history[0] in messages[0]["content"]
    assert action_history[1] in messages[0]["content"]
    # Check results
    assert llm_plan_result.is_goal_complete is True
    assert llm_plan_result.action == "click"
    assert target_element_result is not None
    assert target_element_result.id == 4
