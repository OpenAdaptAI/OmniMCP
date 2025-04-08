# tests/test_core.py
import pytest

# Import types from the project
from omnimcp.types import (
    UIElement,
    Bounds,
    ElementTrack,
    ScreenAnalysis,
    ActionDecision,
    LLMAnalysisAndDecision,
)

# Import the function to test
from omnimcp.core import plan_action_for_ui

# Assuming pytest-mock (mocker fixture) is available


# --- Fixture for Sample Elements (Keep as is) ---
@pytest.fixture
def sample_elements() -> list[UIElement]:
    # ... (fixture definition as provided previously) ...
    bounds_tf1: Bounds = (0.1, 0.1, 0.8, 0.05)
    bounds_tf2: Bounds = (0.1, 0.2, 0.8, 0.05)
    bounds_cb: Bounds = (0.1, 0.3, 0.3, 0.05)
    bounds_link: Bounds = (0.5, 0.3, 0.4, 0.05)
    bounds_btn: Bounds = (0.4, 0.4, 0.2, 0.08)
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
    tracking_info = None

    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # Setup Mock Response (returning LLMAnalysisAndDecision)
    mock_analysis = ScreenAnalysis(
        reasoning="Need username",
        new_elements=[f"track_{i}" for i in range(len(sample_elements))],
        critical_elements_status={"track_0": "Visible"},
    )
    mock_decision = ActionDecision(
        analysis_reasoning="Typing username",
        action_type="type",
        target_element_id=0,
        parameters={"text_to_type": "testuser"},
        is_goal_complete=False,
    )
    mock_combined_output = LLMAnalysisAndDecision(
        screen_analysis=mock_analysis, action_decision=mock_decision
    )
    mock_llm_api.return_value = mock_combined_output

    # Call the function under test - now returns ActionDecision
    action_decision_result, target_element_result = plan_action_for_ui(
        elements=sample_elements,
        user_goal=user_goal,
        action_history=action_history,
        step=step,
        tracking_info=tracking_info,
    )

    # Assertions
    mock_llm_api.assert_called_once()
    call_args, call_kwargs = mock_llm_api.call_args
    assert call_args[1] is LLMAnalysisAndDecision  # Check correct model expected

    prompt_text = call_args[0][0]["content"]
    assert (
        "**Previous Actions Taken (up to last 5):**" in prompt_text
    )  # Corrected assertion
    assert (
        "**Tracked Elements Context (Persistent View - Max 50):**" in prompt_text
    )  # Corrected assertion
    assert "(No tracking info available or first frame)" in prompt_text

    # --- START FIX: Assert against ActionDecision fields ---
    assert isinstance(action_decision_result, ActionDecision)
    assert action_decision_result.action_type == "type"
    assert action_decision_result.target_element_id == 0
    assert action_decision_result.parameters.get("text_to_type") == "testuser"
    assert action_decision_result.parameters.get("key_info") is None
    assert action_decision_result.is_goal_complete is False
    assert (
        action_decision_result.analysis_reasoning == "Typing username"
    )  # Check reasoning part
    # --- END FIX ---
    assert target_element_result is sample_elements[0]


def test_plan_action_step3_click_login(mocker, sample_elements):
    """Test planning the third step: clicking login and completing goal."""
    user_goal = "Log in as testuser with password pass"
    sample_elements[0].content = "testuser"
    sample_elements[1].content = "********"
    action_history = ["Step 1...", "Step 2..."]
    step = 2
    mock_tracking_info = [
        ElementTrack(track_id=f"track_{el.id}", latest_element=el, last_seen_frame=step)
        for el in sample_elements
    ]

    mock_llm_api = mocker.patch("omnimcp.core.call_llm_api")

    # Setup Mock Response (returning LLMAnalysisAndDecision)
    mock_analysis_step3 = ScreenAnalysis(
        reasoning="Ready to click Login.",
        critical_elements_status={"track_4": "Visible"},
    )
    mock_decision_step3 = ActionDecision(
        analysis_reasoning="Clicking Login button.",
        action_type="click",
        target_element_id=4,
        parameters={},
        is_goal_complete=True,
    )
    mock_combined_output_step3 = LLMAnalysisAndDecision(
        screen_analysis=mock_analysis_step3, action_decision=mock_decision_step3
    )
    mock_llm_api.return_value = mock_combined_output_step3

    # Call the function
    action_decision_result, target_element_result = plan_action_for_ui(
        elements=sample_elements,
        user_goal=user_goal,
        action_history=action_history,
        step=step,
        tracking_info=mock_tracking_info,
    )

    # Assertions
    mock_llm_api.assert_called_once()
    call_args, call_kwargs = mock_llm_api.call_args
    assert call_args[1] is LLMAnalysisAndDecision

    prompt_text = call_args[0][0]["content"]
    assert action_history[0] in prompt_text
    assert action_history[1] in prompt_text
    assert (
        "**Tracked Elements Context (Persistent View - Max 50):**" in prompt_text
    )  # Corrected assertion
    assert "TrackID track_4" in prompt_text
    assert "Status: VISIBLE" in prompt_text

    # --- START FIX: Assert against ActionDecision fields ---
    assert isinstance(action_decision_result, ActionDecision)
    assert action_decision_result.is_goal_complete is True
    assert action_decision_result.action_type == "click"
    assert action_decision_result.target_element_id == 4
    assert action_decision_result.parameters == {}
    assert action_decision_result.analysis_reasoning == "Clicking Login button."
    # --- END FIX ---
    assert target_element_result is sample_elements[4]
