# tests/test_core.py
import pytest

# Import types from the project
from omnimcp.types import (
    UIElement,
    Bounds,
    LLMActionPlan,
    ElementTrack,
    ScreenAnalysis,
    ActionDecision,
    LLMAnalysisAndDecision,  # Added new types
)

# Import the function to test
from omnimcp.core import plan_action_for_ui

# Assuming pytest-mock (mocker fixture) is available

# --- Fixture for Sample Elements ---


@pytest.fixture
def sample_elements() -> list[UIElement]:
    """Provides a sample list of UIElements similar to the login screen."""
    # Use slightly more distinct bounds for testing
    bounds_tf1: Bounds = (0.1, 0.1, 0.8, 0.05)
    bounds_tf2: Bounds = (0.1, 0.2, 0.8, 0.05)  # Below first field
    bounds_cb: Bounds = (0.1, 0.3, 0.3, 0.05)
    bounds_link: Bounds = (0.5, 0.3, 0.4, 0.05)
    bounds_btn: Bounds = (0.4, 0.4, 0.2, 0.08)  # Centered below

    return [
        UIElement(
            id=0,
            type="text_field",
            content="",
            bounds=bounds_tf1,
            attributes={"label": "Username:"},
            confidence=0.95,
        ),
        UIElement(
            id=1,
            type="text_field",
            content="",
            bounds=bounds_tf2,
            attributes={"is_password": True, "label": "Password:"},
            confidence=0.95,
        ),
        UIElement(
            id=2,
            type="checkbox",
            content="Remember Me",
            bounds=bounds_cb,
            attributes={"checked": False},
            confidence=0.90,
        ),
        UIElement(
            id=3,
            type="link",
            content="Forgot Password?",
            bounds=bounds_link,
            confidence=0.92,
        ),
        UIElement(
            id=4, type="button", content="Login", bounds=bounds_btn, confidence=0.98
        ),
    ]


# --- Tests for plan_action_for_ui ---


def test_plan_action_step1_type_user(mocker, sample_elements):
    """Test planning the first step: typing username."""
    user_goal = "Log in as testuser with password pass"
    action_history = []
    step = 0
    tracking_info = None  # No tracking info on first step

    # Mock the LLM API call within the core module
    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # --- Setup Mock Response ---
    # Configure the mock to return the NEW structure (LLMAnalysisAndDecision)
    mock_analysis = ScreenAnalysis(
        reasoning="Goal is log in. History empty. Need username in field ID 0.",
        new_elements=[
            f"track_{i}" for i in range(len(sample_elements))
        ],  # Assume all new
        critical_elements_status={"track_0": "Visible"},
    )
    mock_decision = ActionDecision(
        analysis_reasoning="Typing username into field ID 0.",
        action_type="type",
        target_element_id=0,  # Target the first text field (ID 0)
        parameters={"text_to_type": "testuser"},
        is_goal_complete=False,
    )
    mock_combined_output = LLMAnalysisAndDecision(
        screen_analysis=mock_analysis, action_decision=mock_decision
    )
    mock_llm_api.return_value = mock_combined_output
    # --- End Mock Response Setup ---

    # Call the function under test
    # plan_action_for_ui internally converts result back to LLMActionPlan for now
    llm_plan_result, target_element_result = plan_action_for_ui(
        elements=sample_elements,
        user_goal=user_goal,
        action_history=action_history,
        step=step,
        tracking_info=tracking_info,
    )

    # --- Assertions ---
    mock_llm_api.assert_called_once()
    call_args, call_kwargs = mock_llm_api.call_args
    # Check arguments passed to call_llm_api
    messages = call_args[0]
    response_model_passed = call_args[1]
    assert (
        response_model_passed is LLMAnalysisAndDecision
    )  # Check correct model was expected

    # Check basic prompt content
    prompt_text = messages[0]["content"]
    assert user_goal in prompt_text
    assert sample_elements[0].to_prompt_repr() in prompt_text
    assert "**Previous Actions Taken" in prompt_text
    assert "**Tracked Elements Context" in prompt_text
    assert (
        "(No tracking info available or first frame)" in prompt_text
    )  # Check tracking section rendering

    # Check returned values (should be converted LLMActionPlan)
    assert isinstance(llm_plan_result, LLMActionPlan)
    assert llm_plan_result.action == "type"
    assert llm_plan_result.element_id == 0
    assert llm_plan_result.text_to_type == "testuser"
    assert llm_plan_result.key_info is None
    assert llm_plan_result.is_goal_complete is False
    assert "Typing username into field ID 0" in llm_plan_result.reasoning
    assert target_element_result is sample_elements[0]  # Check correct element returned


def test_plan_action_step3_click_login(mocker, sample_elements):
    """Test planning the third step: clicking login and completing goal."""
    user_goal = "Log in as testuser with password pass"
    # Simulate state where fields are filled (by updating content)
    sample_elements[0].content = "testuser"  # Username field filled
    sample_elements[1].content = "********"  # Password field filled (masked)
    action_history = [
        "Step 1: Planned type on ElemID 0 Text='testuser'",
        "Step 2: Planned type on ElemID 1 Text='********'",
    ]
    step = 2  # 3rd step (0-indexed)
    # Simulate tracking info (assume all elements are persistent and visible)
    mock_tracking_info = [
        ElementTrack(track_id=f"track_{el.id}", latest_element=el, last_seen_frame=step)
        for el in sample_elements
    ]

    # Mock the LLM API call
    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # --- Setup Mock Response ---
    mock_analysis_step3 = ScreenAnalysis(
        reasoning="Username and password seem entered based on history. Login button (TrackID track_4) is visible. Ready to click.",
        critical_elements_status={"track_4": "Visible"},
        # Assume no new/disappeared elements for simplicity in this mock
    )
    mock_decision_step3 = ActionDecision(
        analysis_reasoning="Clicking Login button to attempt login.",
        action_type="click",
        target_element_id=4,  # Target the Login button (ID 4)
        parameters={},
        is_goal_complete=True,  # Assume LLM thinks goal completes after click
    )
    mock_combined_output_step3 = LLMAnalysisAndDecision(
        screen_analysis=mock_analysis_step3, action_decision=mock_decision_step3
    )
    mock_llm_api.return_value = mock_combined_output_step3
    # --- End Mock Response Setup ---

    # Call the function
    llm_plan_result, target_element_result = plan_action_for_ui(
        elements=sample_elements,
        user_goal=user_goal,
        action_history=action_history,
        step=step,
        tracking_info=mock_tracking_info,  # Pass the mock tracking info
    )

    # --- Assertions ---
    mock_llm_api.assert_called_once()
    call_args, call_kwargs = mock_llm_api.call_args
    messages = call_args[0]
    response_model_passed = call_args[1]
    assert response_model_passed is LLMAnalysisAndDecision

    # Check history and tracking rendering in prompt
    prompt_text = messages[0]["content"]
    assert action_history[0] in prompt_text
    assert action_history[1] in prompt_text
    assert "**Tracked Elements Context" in prompt_text
    assert "TrackID track_4" in prompt_text  # Check a specific track mentioned
    assert "Status: VISIBLE" in prompt_text  # Check status rendering

    # Check results (converted LLMActionPlan)
    assert isinstance(llm_plan_result, LLMActionPlan)
    assert llm_plan_result.is_goal_complete is True
    assert llm_plan_result.action == "click"
    assert llm_plan_result.element_id == 4
    assert llm_plan_result.text_to_type is None
    assert llm_plan_result.key_info is None
    assert "Clicking Login button" in llm_plan_result.reasoning
    assert target_element_result is sample_elements[4]  # Check correct element returned
