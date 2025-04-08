# tests/test_agent_executor.py

import pytest
import os  # Import os
from unittest.mock import MagicMock
from PIL import Image
from typing import List, Optional, Tuple

# Necessary type imports
from omnimcp.types import (
    UIElement,
    ElementTrack,
    ActionDecision,
)

# Imports from the module under test
from omnimcp.agent_executor import AgentExecutor, PlannerCallable

# Import module itself for patching module-level functions
from omnimcp import agent_executor
from loguru import logger


# --- Mock Execution Class ---
class MockExecution:
    """Mocks the ExecutionInterface."""

    def __init__(self):
        self.calls = []
        self.fail_on_action: Optional[str] = None

    def click(self, x: int, y: int, click_type: str = "single") -> bool:
        self.calls.append(("click", x, y, click_type))
        logger.debug(f"MockExecution: click({x}, {y}, '{click_type}')")
        return self.fail_on_action != "click"

    def type_text(self, text: str) -> bool:
        self.calls.append(("type_text", text))
        logger.debug(f"MockExecution: type_text('{text[:50]}...')")
        return self.fail_on_action != "type"

    def execute_key_string(self, key_info_str: str) -> bool:
        self.calls.append(("execute_key_string", key_info_str))
        logger.debug(f"MockExecution: execute_key_string('{key_info_str}')")
        return self.fail_on_action != "press_key"

    def scroll(self, dx: int, dy: int) -> bool:
        self.calls.append(("scroll", dx, dy))
        logger.debug(f"MockExecution: scroll({dx}, {dy})")
        return self.fail_on_action != "scroll"


# --- Mock Perception Class ---
class MockPerception:
    """Mocks the PerceptionInterface for testing AgentExecutor."""

    elements: List[UIElement]
    tracked_elements_view: List[ElementTrack]
    screen_dimensions: Optional[Tuple[int, int]]
    _last_screenshot: Optional[Image.Image]
    frame_counter: int
    update_call_count: int  # Correct attribute name
    fail_on_update: bool = False

    def __init__(
        self,
        elements_to_return: Optional[List[UIElement]] = None,
        dims: Optional[Tuple[int, int]] = (200, 100),
    ):
        """Initializes the mock perception component."""
        self.elements_to_return = (
            elements_to_return if elements_to_return is not None else []
        )
        self.screen_dimensions = dims
        self.elements = []
        self.tracked_elements_view = []
        self.frame_counter = 0
        self._last_screenshot = Image.new("RGB", dims) if dims else None
        self.update_call_count = 0  # Initialize correctly
        self.fail_on_update = False
        logger.debug("MockPerception initialized.")

    def update(self) -> None:
        """Simulates updating the perception state."""
        self.update_call_count += 1  # Increment correct counter
        self.frame_counter += 1

        if self.fail_on_update and self.update_call_count > 1:
            logger.error("MockPerception: Simulating perception failure.")
            raise RuntimeError("Simulated perception failure")

        self.elements = self.elements_to_return
        self.tracked_elements_view = []  # Mock returns empty tracking view
        if self.screen_dimensions:
            self._last_screenshot = Image.new("RGB", self.screen_dimensions)
        else:
            self._last_screenshot = None
        logger.debug(
            f"MockPerception updated (call {self.update_call_count}, frame {self.frame_counter})"
        )


# --- Fixtures ---
@pytest.fixture
def mock_perception_component() -> MockPerception:
    """Provides a default MockPerception instance with one element."""
    return MockPerception(
        elements_to_return=[
            UIElement(
                id=0,
                type="button",
                content="OK",
                bounds=(0.1, 0.1, 0.2, 0.1),
                confidence=1.0,
                attributes={},
            )
        ]
    )


@pytest.fixture
def mock_execution_component() -> MockExecution:
    """Provides a MockExecution instance."""
    return MockExecution()


@pytest.fixture
def mock_element() -> UIElement:
    """Provides a sample UIElement for tests."""
    return UIElement(id=0, type="button", content="OK", bounds=(0.1, 0.1, 0.2, 0.1))


@pytest.fixture
def temp_output_dir(tmp_path) -> str:
    """Creates a temporary directory for test run outputs."""
    run_dir = tmp_path / "test_runs"
    run_dir.mkdir(exist_ok=True)
    return str(run_dir)


@pytest.fixture
def mock_box_drawer() -> MagicMock:
    """Provides a mock for the draw_bounding_boxes utility."""
    return MagicMock(return_value=Image.new("RGB", (10, 10)))


@pytest.fixture
def mock_highlighter() -> MagicMock:
    """Provides a mock for the draw_action_highlight utility."""
    return MagicMock(return_value=Image.new("RGB", (10, 10)))


# --- Mock Planners (Updated to return ActionDecision) ---


def planner_completes_on_step(n: int) -> PlannerCallable:
    """Factory for a planner that completes on step index `n-1`."""

    def mock_planner(
        elements: List[UIElement],
        user_goal: str,
        action_history: List[str],
        step: int,
        tracking_info: Optional[List[ElementTrack]] = None,
    ) -> Tuple[ActionDecision, Optional[UIElement]]:  # Return ActionDecision
        target_element = elements[0] if elements else None
        is_complete = step == (n - 1)  # Complete on index n-1
        action_type = "click" if not is_complete else "finish"  # Use 'finish' action
        target_element_id = (
            target_element.id if target_element and action_type == "click" else None
        )
        params = {}

        decision = ActionDecision(
            analysis_reasoning=f"Mock analysis step {step + 1}. Complete={is_complete}",
            action_type=action_type,
            target_element_id=target_element_id,
            parameters=params,
            is_goal_complete=is_complete,
        )
        logger.debug(
            f"Mock Planner (complete on {n}): Step {step}, Returning ActionDecision: {decision.action_type}"
        )
        return decision, target_element

    return mock_planner


def planner_never_completes() -> PlannerCallable:
    """Factory for a planner that never signals goal completion."""

    def mock_planner(
        elements: List[UIElement],
        user_goal: str,
        action_history: List[str],
        step: int,
        tracking_info: Optional[List[ElementTrack]] = None,
    ) -> Tuple[ActionDecision, Optional[UIElement]]:  # Return ActionDecision
        target_element = elements[0] if elements else None
        target_element_id = target_element.id if target_element else None

        decision = ActionDecision(
            analysis_reasoning=f"Mock analysis step {step + 1}, goal not complete.",
            action_type="click",  # Always plans click
            target_element_id=target_element_id,
            parameters={},
            is_goal_complete=False,
        )
        logger.debug(
            f"Mock Planner (never complete): Step {step}, Returning ActionDecision: {decision.action_type}"
        )
        return decision, target_element

    return mock_planner


def planner_fails() -> PlannerCallable:
    """Factory for a planner that raises an exception."""

    def failing_planner(*args, **kwargs):  # Accept any args
        logger.error("Mock Planner: Simulating planning failure.")
        raise ValueError("Mock planning failure")

    return failing_planner  # type: ignore


# --- Test Functions (Updated Assertions) ---


def test_agent_executor_init(mock_perception_component, mock_execution_component):
    """Test basic initialization."""
    planner = MagicMock()
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner,
        execution=mock_execution_component,
    )
    assert executor._perception is mock_perception_component
    assert executor._planner is planner
    assert executor._execution is mock_execution_component
    assert executor.action_history == []
    assert isinstance(executor.metrics, dict)


def test_run_completes_goal(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    mock_box_drawer: MagicMock,
    mock_highlighter: MagicMock,
    temp_output_dir: str,
    mocker,
):
    """Test a successful run completing on the second step (index 1)."""
    mock_final_image = Image.new("RGB", (50, 50), color="green")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )

    complete_step_n = 2  # Completes ON step 2 (index 1)
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_completes_on_step(complete_step_n),
        execution=mock_execution_component,
        box_drawer=mock_box_drawer,
        highlighter=mock_highlighter,
    )

    result = executor.run(
        goal="Test Goal", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is True, "Should return True when goal is completed."
    # Perception called for step 0 and step 1 (Total: 2)
    assert mock_perception_component.update_call_count == complete_step_n
    # Execution called only for step 0 (Click before completion on step 1)
    assert len(mock_execution_component.calls) == complete_step_n - 1
    assert mock_execution_component.calls[0][0] == "click"  # Action in step 0
    # History includes plan for step 0 and step 1 (Total: 2)
    assert len(executor.action_history) == complete_step_n

    # Check output files
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "step_1_state_raw.png"))
    assert os.path.exists(os.path.join(run_dir_path, "step_2_state_raw.png"))
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))
    assert os.path.exists(os.path.join(run_dir_path, "run_metrics.json"))
    assert os.path.exists(os.path.join(run_dir_path, "run_log.jsonl"))
    # Visualizers called for each step before potential break
    assert mock_box_drawer.call_count == complete_step_n
    assert (
        mock_highlighter.call_count == 0
    )  # Highlighter call is currently commented out


def test_run_reaches_max_steps(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    mock_box_drawer: MagicMock,
    mock_highlighter: MagicMock,
    temp_output_dir: str,
    mocker,
):
    """Test reaching max_steps without completing the goal."""
    mock_final_image = Image.new("RGB", (50, 50), color="blue")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )

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
    assert (
        mock_highlighter.call_count == 0
    )  # Highlighter call is currently commented out
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_perception_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,
):
    """Test that the loop stops if perception fails on the second step."""
    mock_final_image = Image.new("RGB", (50, 50), color="red")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )

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
    # Update called twice: first succeeds, second raises exception
    assert mock_perception_component.update_call_count == 2
    # Execution only happens for the first step (step 0)
    assert len(mock_execution_component.calls) == 1
    assert len(executor.action_history) == 1
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_planning_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,
):
    """Test that the loop stops if planning fails."""
    mock_final_image = Image.new("RGB", (50, 50), color="yellow")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )

    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_fails(),  # Use the planner that raises an exception
        execution=mock_execution_component,
    )

    result = executor.run(
        goal="Test Planning Fail", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is False
    assert mock_perception_component.update_call_count == 1
    assert len(mock_execution_component.calls) == 0
    assert len(executor.action_history) == 0
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))


def test_run_execution_failure(
    mock_perception_component: MockPerception,
    mock_execution_component: MockExecution,
    temp_output_dir: str,
    mocker,
):
    """Test that the loop stops if execution fails."""
    mock_final_image = Image.new("RGB", (50, 50), color="purple")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )

    mock_execution_component.fail_on_action = "click"
    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_never_completes(),  # Planner plans 'click' on step 0
        execution=mock_execution_component,
    )

    result = executor.run(
        goal="Test Execution Fail", max_steps=5, output_base_dir=temp_output_dir
    )

    assert result is False
    assert mock_perception_component.update_call_count == 1
    assert len(mock_execution_component.calls) == 1  # Execution was attempted
    assert len(executor.action_history) == 1
    assert executor.action_history[0].startswith("Step 1: Planned click")
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
    mock_final_image = Image.new("RGB", (50, 50), color="orange")
    mocker.patch.object(
        agent_executor, "take_screenshot", return_value=mock_final_image
    )
    mocker.patch.object(
        agent_executor, "get_scaling_factor", return_value=scaling_factor
    )

    # Configure MagicMock planner to return ActionDecision tuple
    mock_decision_for_click = ActionDecision(
        analysis_reasoning="Click test",
        action_type="click",
        target_element_id=mock_element.id,
        parameters={},
        is_goal_complete=False,
    )
    planner_click = MagicMock(
        return_value=(
            mock_decision_for_click,
            mock_element,
        )  # Return ActionDecision tuple
    )

    executor = AgentExecutor(
        perception=mock_perception_component,
        planner=planner_click,
        execution=mock_execution_component,
    )

    executor.run(goal="Test Scaling", max_steps=1, output_base_dir=temp_output_dir)

    # Verify planner call arguments
    planner_click.assert_called_once()
    call_args, call_kwargs = planner_click.call_args
    assert call_kwargs.get("tracking_info") == []  # Check tracking info passed

    # Verify execution call
    # MockPerception dims: W=200, H=100 ; MockElement bounds: x=0.1,y=0.1,w=0.2,h=0.1
    # Center physical x = (0.1 + 0.2 / 2) * 200 = 40
    # Center physical y = (0.1 + 0.1 / 2) * 100 = 15
    expected_logical_x = int(40 / scaling_factor)
    expected_logical_y = int(15 / scaling_factor)

    assert len(mock_execution_component.calls) == 1, (
        "Execution component should have been called once"
    )
    assert mock_execution_component.calls[0] == (
        "click",
        expected_logical_x,
        expected_logical_y,
        "single",
    ), f"Click coordinates incorrect for scaling factor {scaling_factor}"

    # Check output files
    run_dirs = os.listdir(temp_output_dir)
    assert len(run_dirs) == 1, "Expected one run directory"
    run_dir_path = os.path.join(temp_output_dir, run_dirs[0])
    assert os.path.exists(os.path.join(run_dir_path, "final_state.png"))
