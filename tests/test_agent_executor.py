# tests/test_agent_executor.py

import os
from unittest.mock import MagicMock

import pytest
from PIL import Image

from omnimcp import agent_executor
from omnimcp.agent_executor import (
    AgentExecutor,
    ExecutionInterface,
    PerceptionInterface,
    PlannerCallable,
)
from omnimcp.types import LLMActionPlan, UIElement


class MockPerception(PerceptionInterface):
    def __init__(
        self,
        elements: list[UIElement],
        dims: tuple[int, int] | None,
        image: Image.Image | None,
    ):
        self.elements = elements
        self.screen_dimensions = dims
        self._last_screenshot = image
        self.update_call_count = 0
        self.fail_on_update = False  # Flag to simulate failure

    def update(self) -> None:
        if (
            self.fail_on_update and self.update_call_count > 0
        ):  # Fail on second+ call if requested
            raise ConnectionError("Mock perception failure")
        self.update_call_count += 1
        # Simulate state update if needed, or keep static for simple tests


class MockExecution(ExecutionInterface):
    def __init__(self):
        self.calls = []
        self.fail_on_action: str | None = None  # e.g., "click" to make click fail

    def click(self, x: int, y: int, click_type: str = "single") -> bool:
        self.calls.append(("click", x, y, click_type))
        return self.fail_on_action != "click"

    def type_text(self, text: str) -> bool:
        self.calls.append(("type_text", text))
        return self.fail_on_action != "type"

    def execute_key_string(self, key_info_str: str) -> bool:
        self.calls.append(("execute_key_string", key_info_str))
        return self.fail_on_action != "press_key"

    def scroll(self, dx: int, dy: int) -> bool:
        self.calls.append(("scroll", dx, dy))
        return self.fail_on_action != "scroll"


# --- Pytest Fixtures ---


@pytest.fixture
def mock_image() -> Image.Image:
    return Image.new("RGB", (200, 100), color="gray")  # Slightly larger default


@pytest.fixture
def mock_element() -> UIElement:
    return UIElement(id=0, type="button", content="OK", bounds=(0.1, 0.1, 0.2, 0.1))


@pytest.fixture
def mock_perception_component(mock_element, mock_image) -> MockPerception:
    return MockPerception([mock_element], (200, 100), mock_image)


@pytest.fixture
def mock_execution_component() -> MockExecution:
    return MockExecution()


@pytest.fixture
def mock_box_drawer() -> MagicMock:
    return MagicMock(return_value=Image.new("RGB", (1, 1)))  # Return dummy image


@pytest.fixture
def mock_highlighter() -> MagicMock:
    return MagicMock(return_value=Image.new("RGB", (1, 1)))  # Return dummy image


@pytest.fixture
def temp_output_dir(tmp_path) -> str:
    """Create a temporary directory for test run outputs."""
    # tmp_path is a pytest fixture providing a Path object to a unique temp dir
    output_dir = tmp_path / "test_runs"
    output_dir.mkdir()
    return str(output_dir)


# --- Mock Planners ---


def planner_completes_on_step(n: int) -> PlannerCallable:
    """Factory for a planner that completes on step index `n`."""

    def mock_planner(
        elements: list[UIElement], user_goal: str, action_history: list[str], step: int
    ) -> tuple[LLMActionPlan, UIElement | None]:
        target_element = elements[0] if elements else None
        is_complete = step == n
        action = "click" if not is_complete else "press_key"  # Vary action
        element_id = target_element.id if target_element and action == "click" else None
        key_info = "Enter" if is_complete else None

        plan = LLMActionPlan(
            reasoning=f"Mock reasoning step {step + 1} for goal '{user_goal}'",
            action=action,
            element_id=element_id,
            key_info=key_info,
            is_goal_complete=is_complete,
        )
        return plan, target_element

    return mock_planner


def planner_never_completes() -> PlannerCallable:
    """Planner that never signals goal completion."""

    def mock_planner(
        elements: list[UIElement], user_goal: str, action_history: list[str], step: int
    ) -> tuple[LLMActionPlan, UIElement | None]:
        target_element = elements[0] if elements else None
        element_id = target_element.id if target_element else None
        plan = LLMActionPlan(
            reasoning=f"Mock reasoning step {step + 1} for goal '{user_goal}', goal not complete",
            action="click",
            element_id=element_id,
            text_to_type=None,
            key_info=None,
            is_goal_complete=False,
        )
        return plan, target_element

    return mock_planner


def planner_fails() -> PlannerCallable:
    """Planner that raises an exception."""

    def failing_planner(*args, **kwargs):
        raise ValueError("Mock planning failure")

    return failing_planner


# --- Test Functions ---


def test_run_completes_goal(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    mock_box_drawer: MagicMock,
    mock_highlighter: MagicMock,
    temp_output_dir: str,
    mocker,  # Add mocker fixture
):
    """Test a successful run where the goal is completed on the second step."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    mock_final_image = Image.new("RGB", (50, 50), color="green")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    complete_step_index = 1
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_completes_on_step(complete_step_index),
        execution=mock_execution_component,
        box_drawer=mock_box_drawer,
        highlighter=mock_highlighter,
    )

    result = executor.run(
        goal="Test Goal", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is True, "Should return True when goal is completed."
    assert (
        mock_perception_component.update_call_count == complete_step_index + 1
    )  # Called for steps 0, 1
    assert (
        len(mock_execution_component.calls) == complete_step_index
    )  # Executed only for step 0
    assert mock_execution_component.calls[0][0] == "click"  # Action in step 0
    assert len(executor.action_history) == complete_step_index

    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "step_1_state_raw.png"))
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))
    assert mock_box_drawer.call_count == complete_step_index + 1
    assert mock_highlighter.call_count == complete_step_index


def test_run_reaches_max_steps(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    mock_box_drawer: MagicMock,
    mock_highlighter: MagicMock,
    temp_output_dir: str,
    mocker,  # Add mocker fixture for consistency, patch take_screenshot here too
):
    """Test reaching max_steps without completing the goal."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    mock_final_image = Image.new("RGB", (50, 50), color="blue")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    max_steps = 3
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_never_completes(),
        execution=mock_execution_component,
        box_drawer=mock_box_drawer,
        highlighter=mock_highlighter,
    )

    result = executor.run(
        goal="Test Max Steps", max_steps=max_steps, output_base_dir=temp_output_dir
    )

    assert result is False, "Should return False when max steps reached."
    assert mock_perception_component.update_call_count == max_steps
    assert len(mock_execution_component.calls) == max_steps
    assert len(executor.action_history) == max_steps
    assert mock_box_drawer.call_count == max_steps
    assert mock_highlighter.call_count == max_steps
    # Also check final state image existence here
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_perception_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,  # Add mocker fixture
):
    """Test that the loop stops if perception fails on the second step."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    mock_final_image = Image.new("RGB", (50, 50), color="red")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    mock_perception_component.fail_on_update = True  # Configure mock to fail
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_never_completes(),
        execution=mock_execution_component,
    )

    result = executor.run(
        goal="Test Perception Fail", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is False
    assert (
        mock_perception_component.update_call_count == 1
    )  # First call ok, fails during second
    assert len(mock_execution_component.calls) == 1  # Only first step executed
    assert len(executor.action_history) == 1
    # Check final state image existence
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_planning_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,  # Add mocker fixture
):
    """Test that the loop stops if planning fails."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    mock_final_image = Image.new("RGB", (50, 50), color="yellow")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_fails(),
        execution=mock_execution_component,
    )

    result = executor.run(
        goal="Test Planning Fail", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is False
    assert (
        mock_perception_component.update_call_count == 1
    )  # Perception called once before planning
    assert len(mock_execution_component.calls) == 0  # Execution never reached
    # Check final state image existence
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_execution_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,  # Add mocker fixture
):
    """Test that the loop stops if execution fails."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    mock_final_image = Image.new("RGB", (50, 50), color="purple")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    mock_execution_component.fail_on_action = "click"  # Make the click action fail
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_never_completes(),  # Planner plans 'click' first
        execution=mock_execution_component,
    )

    result = executor.run(
        goal="Test Execution Fail", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is False
    assert mock_perception_component.update_call_count == 1
    assert len(mock_execution_component.calls) == 1  # Execution was attempted
    assert executor.action_history[0].startswith(
        "Step 1: Planned click"
    )  # History includes planned action
    # Check final state image existence
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


@pytest.mark.parametrize("scaling_factor", [1, 2])
def test_coordinate_scaling_for_click(
    mock_perception_component: MockPerception,
    mock_element: UIElement,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,
    scaling_factor: int,
):
    """Verify coordinate scaling is applied before calling execution.click."""
    # --- Add Mock for take_screenshot to avoid $DISPLAY error in CI ---
    # (Not strictly necessary here as loop only runs 1 step, but good practice)
    mock_final_image = Image.new("RGB", (50, 50), color="orange")  # Dummy image
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    # --- End Mock ---

    planner_click = MagicMock(
        return_value=(
            LLMActionPlan(
                reasoning="Click test",
                action="click",
                element_id=mock_element.id,
                is_goal_complete=False,
            ),
            mock_element,
        )
    )
    # Patch get_scaling_factor within the agent_executor module
    mocker.patch.object(
        agent_executor, "get_scaling_factor", return_value=scaling_factor
    )

    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_click,
        execution=mock_execution_component,
    )

    executor.run(goal="Test Scaling", max_steps=1, output_base_dir=temp_output_dir)

    # Dims: W=200, H=100
    # Bounds: x=0.1, y=0.1, w=0.2, h=0.1
    # Center physical x = (0.1 + 0.2 / 2) * 200 = 40
    # Center physical y = (0.1 + 0.1 / 2) * 100 = 15
    expected_logical_x = int(40 / scaling_factor)
    expected_logical_y = int(15 / scaling_factor)

    assert len(mock_execution_component.calls) == 1
    assert mock_execution_component.calls[0] == (
        "click",
        expected_logical_x,
        expected_logical_y,
        "single",
    )
    # Check final state image existence
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))
