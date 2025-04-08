# omnimcp/agent_executor.py

import datetime
import os
import time
from typing import Callable, List, Optional, Tuple, Protocol, Dict, Any
import json

from PIL import Image
from loguru import logger

# Local imports using relative paths within the package
from . import config

from .types import (
    UIElement,
    ElementTrack,
    LoggedStep,
    ActionDecision,
)
from .utils import (
    denormalize_coordinates,
    draw_action_highlight,
    draw_bounding_boxes,
    get_scaling_factor,
    take_screenshot,
    setup_run_logging,
)

# --- Interface Definitions ---


class PerceptionInterface(Protocol):
    """Defines the expected interface for the perception component."""

    elements: List[UIElement]
    tracked_elements_view: List[ElementTrack]
    screen_dimensions: Optional[Tuple[int, int]]
    _last_screenshot: Optional[Image.Image]
    frame_counter: int

    def update(self) -> None: ...


class ExecutionInterface(Protocol):
    """Defines the expected interface for the execution component."""

    def click(self, x: int, y: int, click_type: str = "single") -> bool: ...
    def type_text(self, text: str) -> bool: ...
    def execute_key_string(self, key_info_str: str) -> bool: ...
    def scroll(self, dx: int, dy: int) -> bool: ...


# PlannerCallable expects ActionDecision as the primary return type now
PlannerCallable = Callable[
    [  # Inputs:
        List[UIElement],
        str,
        List[str],
        int,
        Optional[List[ElementTrack]],
    ],
    # Outputs:
    Tuple[ActionDecision, Optional[UIElement]],
]

ImageProcessorCallable = Callable[..., Image.Image]

# --- Core Agent Executor ---


class AgentExecutor:
    """
    Orchestrates the perceive-plan-act loop, integrating perception with tracking,
    planning (using ActionDecision), execution, and structured logging.
    """

    def __init__(
        self,
        perception: PerceptionInterface,
        planner: PlannerCallable,
        execution: ExecutionInterface,
        box_drawer: Optional[ImageProcessorCallable] = draw_bounding_boxes,
        highlighter: Optional[ImageProcessorCallable] = draw_action_highlight,
    ):
        """Initializes the AgentExecutor."""
        self._perception = perception
        self._planner = planner
        self._execution = execution
        self._box_drawer = box_drawer
        self._highlighter = highlighter  # Visualizer for planned action
        self.action_history: List[str] = []

        # Map action names to their handler methods, including new actions
        self._action_handlers: Dict[str, Callable[..., bool]] = {
            "click": self._execute_click,
            "type": self._execute_type,
            "press_key": self._execute_press_key,
            "scroll": self._execute_scroll,
            "wait": self._execute_wait,
            "finish": self._execute_finish,
        }
        # Initialize metrics and structured log storage
        self.metrics: Dict[str, List[Any]] = self._reset_metrics()
        self.run_log_data: List[Dict] = []  # Stores LoggedStep data as dicts
        logger.info("AgentExecutor initialized with updated action handlers.")

    def _reset_metrics(self) -> Dict[str, List[Any]]:
        """Helper to initialize/reset metrics dictionary for a run."""
        return {
            "step_times_s": [],
            "perception_times_s": [],
            "planning_times_s": [],
            "execution_times_s": [],
            "elements_per_step": [],
            "active_tracks_per_step": [],
            "action_results": [],  # Boolean success/fail
        }

    # --- Private Action Handlers (Updated to use ActionDecision) ---

    def _execute_click(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'click' action based on ActionDecision."""
        if not target_element:
            # The planner should have found the element if target_element_id was set.
            # If it's None here, the planner failed to find the element ID specified in the decision.
            logger.error(
                f"Click planned for ElemID {decision.target_element_id} but element could not be resolved in current frame."
            )
            return False
        if not screen_dims:
            logger.error("Cannot execute click without screen dimensions.")
            return False

        abs_x, abs_y = denormalize_coordinates(
            target_element.bounds[0],
            target_element.bounds[1],
            screen_dims[0],
            screen_dims[1],
            target_element.bounds[2],
            target_element.bounds[3],
        )
        logical_x = int(abs_x / scaling_factor)
        logical_y = int(abs_y / scaling_factor)
        click_type = decision.parameters.get(
            "click_type", "single"
        )  # Get optional param
        logger.debug(
            f"Executing {click_type} click at logical coords: ({logical_x}, {logical_y}) on Element ID {target_element.id}"
        )
        return self._execution.click(logical_x, logical_y, click_type=click_type)

    def _execute_type(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'type' action based on ActionDecision."""
        text_to_type = decision.parameters.get("text_to_type")
        if text_to_type is None:  # Check for None specifically, empty string is allowed
            logger.error(
                "Action 'type' planned but 'text_to_type' missing in parameters."
            )
            return False

        # Optional: Click target element first if specified by target_element_id and found
        if target_element and screen_dims:
            abs_x, abs_y = denormalize_coordinates(
                target_element.bounds[0],
                target_element.bounds[1],
                screen_dims[0],
                screen_dims[1],
                target_element.bounds[2],
                target_element.bounds[3],
            )
            logical_x = int(abs_x / scaling_factor)
            logical_y = int(abs_y / scaling_factor)
            logger.debug(
                f"Clicking target Element ID {target_element.id} before typing..."
            )
            if not self._execution.click(logical_x, logical_y):
                logger.warning(
                    "Failed to click target before typing, attempting type anyway."
                )
            time.sleep(0.2)  # Short pause after potential click

        logger.debug(f"Executing type: '{text_to_type[:50]}...'")
        return self._execution.type_text(text_to_type)

    def _execute_press_key(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'press_key' action based on ActionDecision."""
        key_info = decision.parameters.get("key_info")
        if not key_info:
            logger.error(
                "Action 'press_key' planned but 'key_info' missing in parameters."
            )
            return False
        logger.debug(f"Executing press_key: '{key_info}'")
        return self._execution.execute_key_string(key_info)

    def _execute_scroll(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'scroll' action based on ActionDecision."""
        # Attempt to get scroll details from parameters first
        dx = decision.parameters.get("scroll_dx", 0)
        dy = decision.parameters.get("scroll_dy", 0)
        scroll_dir = decision.parameters.get("scroll_direction", "").lower()
        scroll_steps = decision.parameters.get("scroll_steps", 3)

        # Fallback to reasoning hint if parameters are missing
        if dx == 0 and dy == 0 and not scroll_dir:
            scroll_dir_reasoning = decision.analysis_reasoning.lower()
            if "down" in scroll_dir_reasoning:
                scroll_dy = -scroll_steps
            elif "up" in scroll_dir_reasoning:
                scroll_dy = scroll_steps
            if "left" in scroll_dir_reasoning:
                scroll_dx = -scroll_steps
            elif "right" in scroll_dir_reasoning:
                scroll_dx = scroll_steps
        elif scroll_dir:  # Handle direction string if provided
            if "down" in scroll_dir:
                scroll_dy = -scroll_steps
            elif "up" in scroll_dir:
                scroll_dy = scroll_steps
            if "left" in scroll_dir:
                scroll_dx = -scroll_steps
            elif "right" in scroll_dir:
                scroll_dx = scroll_steps

        if scroll_dx != 0 or scroll_dy != 0:
            logger.debug(f"Executing scroll: dx={scroll_dx}, dy={scroll_dy}")
            return self._execution.scroll(scroll_dx, scroll_dy)
        else:
            logger.warning(
                "Scroll planned but direction/amount unclear, skipping scroll."
            )
            return True  # No action needed counts as success

    def _execute_wait(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'wait' action."""
        wait_duration = decision.parameters.get("wait_duration_s", 1.0)  # Default 1s
        try:
            wait_duration = float(wait_duration)
            if wait_duration < 0:
                wait_duration = 0
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid wait_duration '{wait_duration}', defaulting to 1.0s."
            )
            wait_duration = 1.0
        # Define a reasonable maximum wait to prevent infinite loops
        max_wait = 30.0
        wait_duration = min(wait_duration, max_wait)
        logger.info(f"Executing wait for {wait_duration:.1f} seconds...")
        time.sleep(wait_duration)
        return True

    def _execute_finish(
        self,
        decision: ActionDecision,
        target_element: Optional[UIElement],
        screen_dims: Optional[Tuple[int, int]],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'finish' action (no-op, loop breaks)."""
        logger.info(
            "Executing finish action planned by LLM (indicates goal met or stuck)."
        )
        # The main loop checks the is_goal_complete flag from the decision.
        return True  # The action itself succeeds trivially

    # --- Main Execution Loop ---

    # This `run` method implements an explicit, sequential perceive-plan-act loop.
    # Alternative agent architectures exist, such as:
    # - ReAct (Reasoning-Acting): Where the LLM explicitly decides between
    #   reasoning steps and action steps.
    # - Callback-driven: Where UI events or timers might trigger agent actions.
    # - More complex state machines or graph-based execution flows.
    # This simple sequential loop provides a clear baseline. Future work might explore
    # these alternatives for more complex or reactive tasks.
    def run(
        self, goal: str, max_steps: int = 10, output_base_dir: Optional[str] = None
    ) -> bool:
        """Runs the main perceive-plan-act loop to achieve the goal."""
        run_output_dir, log_path = self._setup_run(goal, output_base_dir)
        if not run_output_dir:
            return False  # Exit if setup failed

        self.action_history = []
        self.metrics = self._reset_metrics()
        self.run_log_data = []
        goal_achieved = False
        final_step_success = True  # Tracks if any step failed critically
        last_step_completed = -1
        scaling_factor = 1
        try:
            scaling_factor = get_scaling_factor()
            logger.info(f"Using display scaling factor: {scaling_factor}")
        except Exception as e:
            logger.error(f"Failed to get scaling factor: {e}. Assuming 1.")

        # --- Main Loop ---
        for step in range(max_steps):
            step_start_time = time.time()
            logger.info(f"\n--- Step {step + 1}/{max_steps} ---")
            step_img_prefix = f"step_{step + 1}"

            # Initialize Step Variables
            current_image: Optional[Image.Image] = None
            current_elements: List[UIElement] = []
            tracked_elements_view: List[ElementTrack] = []
            screen_dimensions: Optional[Tuple[int, int]] = None
            tracking_info_for_log: Optional[List[Dict]] = None
            perception_duration = 0.0
            action_decision: Optional[ActionDecision] = None  # Use ActionDecision type
            analysis_log: Optional[Dict] = None  # For logging analysis part
            decision_log: Optional[Dict] = None  # For logging decision part
            target_element: Optional[UIElement] = None
            planning_duration = 0.0
            action_success = False  # Result of the current step's action
            executed_action_type = "none"
            executed_params: Dict[str, Any] = {}
            executed_target_id: Optional[int] = None
            execution_duration = 0.0
            step_screenshot_path: Optional[str] = None

            # 1. Perceive State & Update Tracking
            perception_start_time = time.time()
            try:
                logger.debug("Updating visual state and tracking...")
                self._perception.update()  # Assumes this updates internal tracker

                current_elements = self._perception.elements
                tracked_elements_view = self._perception.tracked_elements_view
                current_image = self._perception._last_screenshot
                screen_dimensions = self._perception.screen_dimensions
                perception_duration = time.time() - perception_start_time

                if not current_image or not screen_dimensions:
                    raise RuntimeError(
                        "Perception failed: Missing image or dimensions."
                    )

                logger.info(
                    f"Perceived state: {len(current_elements)} raw elements, "
                    f"{len(tracked_elements_view)} active tracks. "
                    f"Time: {perception_duration:.2f}s."
                )
                try:
                    tracking_info_for_log = [
                        t.model_dump(mode="json") for t in tracked_elements_view
                    ]
                except Exception as track_dump_err:
                    logger.warning(
                        f"Could not serialize tracking info for log: {track_dump_err}"
                    )
                    tracking_info_for_log = [{"error": "serialization failed"}]

            except Exception as perceive_e:
                logger.error(f"Perception failed: {perceive_e}", exc_info=True)
                final_step_success = False
                perc_time = round(time.time() - perception_start_time, 3)
                self.metrics["perception_times_s"].append(perc_time)
                self.metrics["elements_per_step"].append(0)
                self.metrics["active_tracks_per_step"].append(0)
                step_duration = time.time() - step_start_time
                self._log_step_data(
                    step,
                    goal,
                    None,
                    [],
                    None,
                    None,
                    None,
                    "perception_error",
                    None,
                    {},
                    False,
                    perc_time,
                    0.0,
                    0.0,
                    step_duration,
                )
                break  # Stop run

            self.metrics["perception_times_s"].append(round(perception_duration, 3))
            self.metrics["elements_per_step"].append(len(current_elements))
            self.metrics["active_tracks_per_step"].append(len(tracked_elements_view))

            # 2. Save State Artifacts
            raw_state_path = os.path.join(
                run_output_dir, f"{step_img_prefix}_state_raw.png"
            )
            step_screenshot_path = (
                os.path.relpath(raw_state_path, start=run_output_dir)
                if run_output_dir
                else raw_state_path
            )
            try:
                if current_image:
                    current_image.save(raw_state_path)
                    if self._box_drawer:
                        parsed_state_path = os.path.join(
                            run_output_dir, f"{step_img_prefix}_state_parsed.png"
                        )
                        try:
                            img_with_boxes = self._box_drawer(
                                current_image,
                                current_elements,
                                color="lime",
                                show_ids=True,
                            )
                            img_with_boxes.save(parsed_state_path)
                        except Exception as draw_e:
                            logger.warning(
                                f"Could not save parsed state image: {draw_e}"
                            )
                else:
                    step_screenshot_path = None
            except Exception as save_e:
                logger.warning(f"Could not save state image(s): {save_e}")
                step_screenshot_path = None

            # 3. Plan Action
            planning_start_time = time.time()
            try:
                logger.debug("Planning next action...")
                # Planner now returns ActionDecision and target element
                action_decision, target_element = self._planner(
                    elements=current_elements,
                    user_goal=goal,
                    action_history=self.action_history,
                    step=step,
                    tracking_info=tracked_elements_view,
                )
                planning_duration = time.time() - planning_start_time
                logger.info(f"Planning completed in {planning_duration:.2f}s.")

                if not action_decision:
                    raise ValueError("Planner returned None for ActionDecision")

                # Log details & set execution vars from ActionDecision
                logger.info(
                    f"LLM Decision: Action={action_decision.action_type}, TargetElemID={action_decision.target_element_id}, Params={action_decision.parameters}, GoalComplete={action_decision.is_goal_complete}"
                )
                logger.info(
                    f"LLM Analysis Reasoning: {action_decision.analysis_reasoning}"
                )
                executed_action_type = action_decision.action_type
                executed_target_id = (
                    action_decision.target_element_id
                )  # Current frame ID
                executed_params = action_decision.parameters or {}
                # Store dict representation for logging
                decision_log = action_decision.model_dump(mode="json")
                # analysis_log would come from the ScreenAnalysis part if returned separately by planner

            except Exception as plan_e:
                logger.error(f"Planning failed: {plan_e}", exc_info=True)
                final_step_success = False
                plan_time = round(time.time() - planning_start_time, 3)
                self.metrics["planning_times_s"].append(plan_time)
                step_duration = time.time() - step_start_time
                self._log_step_data(
                    step,
                    goal,
                    step_screenshot_path,
                    current_elements,
                    tracking_info_for_log,
                    None,
                    None,
                    "planning_error",
                    None,
                    {},
                    False,
                    perception_duration,
                    plan_time,
                    0.0,
                    step_duration,
                )
                break

            self.metrics["planning_times_s"].append(round(planning_duration, 3))

            # 4. Check Goal Completion (use ActionDecision)
            if action_decision.is_goal_complete:
                logger.success("LLM determined the goal is achieved!")
                goal_achieved = True

            # 5. Validate Action Requirements (use ActionDecision)
            if (
                not goal_achieved
                and action_decision.action_type == "click"
                and target_element is None
            ):
                logger.error(
                    f"Action 'click' planned for element ID {action_decision.target_element_id}, but planner did not find element. Stopping."
                )
                final_step_success = False
                # Log step data before breaking loop

            # 6. Visualize Planned Action (TODO: Needs update for ActionDecision)
            if self._highlighter and current_image and action_decision:
                # highlight_img_path = os.path.join(
                #     run_output_dir, f"{step_img_prefix}_action_highlight.png"
                # )
                # try:
                #     # Needs draw_action_highlight updated to accept ActionDecision
                #     # highlighted_image = self._highlighter(
                #     #     current_image, element=target_element, plan=action_decision, color="red", width=3
                #     # )
                #     # highlighted_image.save(highlight_img_path)
                #     logger.debug("Skipping action highlight visualization until updated for ActionDecision.")
                # except Exception as draw_highlight_e:
                #     logger.warning(f"Could not save action visualization image: {draw_highlight_e}")
                pass  # Skip highlighting for now

            # 7. Update Action History (use ActionDecision)
            action_desc = f"Step {step + 1}: Planned {action_decision.action_type}"
            if target_element:
                action_desc += f" on ElemID {target_element.id} ('{target_element.content[:20]}...')"
            elif executed_target_id is not None:
                action_desc += f" on ElemID {executed_target_id} (not found)"
            if "text_to_type" in executed_params:
                action_desc += f" Text='{executed_params['text_to_type'][:20]}...'"
            if "key_info" in executed_params:
                action_desc += f" Key='{executed_params['key_info']}'"
            if "wait_duration_s" in executed_params:
                action_desc += f" Wait={executed_params['wait_duration_s']}s"
            self.action_history.append(action_desc)
            logger.debug(f"Added to history: {action_desc}")

            # 8. Execute Action (if needed and possible)
            execution_start_time = time.time()
            if not goal_achieved and final_step_success:
                logger.info(f"Executing action: {executed_action_type}...")
                try:
                    handler = self._action_handlers.get(executed_action_type)
                    if handler:
                        # Pass ActionDecision to handlers
                        action_success = handler(
                            decision=action_decision,
                            target_element=target_element,
                            screen_dims=screen_dimensions,
                            scaling_factor=scaling_factor,
                        )
                    else:
                        logger.error(
                            f"Execution handler for '{executed_action_type}' not found."
                        )
                        action_success = False

                    if not action_success:
                        logger.error(
                            f"Action '{executed_action_type}' execution failed."
                        )
                        final_step_success = False  # Mark run as failed
                    else:
                        logger.success("Action executed successfully.")

                except Exception as exec_e:
                    logger.error(
                        f"Exception during action execution: {exec_e}", exc_info=True
                    )
                    action_success = False
                    final_step_success = False  # Mark run as failed
            else:
                action_success = True  # Treat skipped step as 'successful' non-action
                logger.info(f"Skipping execution for step {step + 1}.")
            execution_duration = time.time() - execution_start_time

            # --- Log Metrics & Step Data ---
            self.metrics["execution_times_s"].append(round(execution_duration, 3))
            self.metrics["action_results"].append(action_success)
            step_duration = time.time() - step_start_time
            self.metrics["step_times_s"].append(round(step_duration, 3))
            self._log_step_data(  # Log full step data
                step,
                goal,
                step_screenshot_path,
                current_elements,
                tracking_info_for_log,
                analysis_log,
                decision_log,  # Pass logs
                executed_action_type,
                executed_target_id,
                executed_params,
                action_success,
                perception_duration,
                planning_duration,
                execution_duration,
                step_duration,
            )

            # --- Check Termination Conditions ---
            if goal_achieved or not final_step_success:
                last_step_completed = step
                logger.info(
                    f"Run ending at step {step + 1} (Goal achieved: {goal_achieved}, Step Success: {final_step_success})"
                )
                break  # Exit the loop

            last_step_completed = step  # Mark step completed if loop continues

            # Wait for UI to settle
            time.sleep(1.0)  # Make configurable later

        # --- End of Loop ---
        logger.info("\n--- Agent Run Finished ---")
        if goal_achieved:
            logger.success("Overall goal marked as achieved.")
        elif not final_step_success:
            logger.error(f"Run failed critically at Step {last_step_completed + 1}.")
        else:
            logger.warning(
                f"Run finished after {max_steps} steps without achieving goal."
            )

        # Save final metrics and logs
        self._save_run_outputs(
            run_output_dir, goal_achieved, final_step_success, last_step_completed
        )

        # Capture final screen state
        try:
            final_state_img_path = os.path.join(run_output_dir, "final_state.png")
            final_image = take_screenshot()
            if final_image:
                final_image.save(final_state_img_path)
                logger.info(f"Saved final screen state to {final_state_img_path}")
        except Exception as save_final_e:
            logger.warning(f"Could not save final state image: {save_final_e}")

        return goal_achieved

    def _setup_run(
        self, goal: str, output_base_dir: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Sets up directories and logging for a new run."""
        if output_base_dir is None:
            output_base_dir = config.RUN_OUTPUT_DIR
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_base_dir, run_timestamp)
        log_path = None  # Initialize
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            # Call the utility function to configure run-specific logging
            log_path = setup_run_logging(
                run_dir=run_output_dir
            )  # Pass the specific dir
            logger.info(f"Starting agent run. Goal: '{goal}'")
            logger.info(f"Saving outputs to: {run_output_dir}")
            if log_path:
                logger.info(f"Run log file: {log_path}")
            return run_output_dir, log_path
        except Exception as setup_e:
            logger.critical(f"Failed during run setup (directory/logging): {setup_e}")
            return None, None  # Return None tuple on failure

    def _log_step_data(
        self,
        step_index,
        goal,
        screenshot_path,
        elements,
        tracking_context,
        analysis_log,
        decision_log,  # Expect dicts
        exec_action,
        exec_target_id,
        exec_params,
        success,
        perc_time,
        plan_time,
        exec_time,
        step_time,
    ):
        """Helper to create and store the structured log entry using LoggedStep."""
        try:
            # We pass decision_log which is already a dict from model_dump
            # analysis_log is currently None, raw_llm_action_plan is None
            step_log_entry = LoggedStep(
                step_index=step_index,
                goal=goal,
                screenshot_path=screenshot_path,
                input_elements_count=len(elements),
                tracking_context=tracking_context,
                action_history_at_step=list(self.action_history),  # Use current history
                llm_analysis=analysis_log,  # Placeholder for now
                llm_decision=decision_log,  # Log the ActionDecision dict
                raw_llm_action_plan=None,  # No longer using this format
                executed_action=exec_action,
                executed_target_element_id=exec_target_id,
                executed_parameters=exec_params,
                action_success=success,
                perception_time_s=round(perc_time, 3),
                planning_time_s=round(plan_time, 3),
                execution_time_s=round(exec_time, 3),
                step_time_s=round(step_time, 3),
            )
            # Append the dictionary representation to the list
            self.run_log_data.append(step_log_entry.model_dump(mode="json"))
        except Exception as log_err:
            logger.warning(
                f"Failed to create/store structured log for step {step_index + 1}: {log_err}",
                exc_info=True,
            )

    def _save_run_outputs(
        self, run_output_dir, goal_achieved, final_step_success, last_step_completed
    ):
        """Helper to save metrics and structured log data at the end of a run."""
        # Save Metrics
        metrics_path = os.path.join(run_output_dir, "run_metrics.json")
        try:
            # Calculate summary stats safely handling potential empty lists
            metrics_details = self.metrics
            valid_step_times = [
                t
                for t in metrics_details["step_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_perc_times = [
                t
                for t in metrics_details["perception_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_plan_times = [
                t
                for t in metrics_details["planning_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_exec_times = [
                t
                for t in metrics_details["execution_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_elem_counts = [
                c for c in metrics_details["elements_per_step"] if isinstance(c, int)
            ]
            valid_track_counts = [
                c
                for c in metrics_details["active_tracks_per_step"]
                if isinstance(c, int)
            ]

            summary_metrics = {
                "total_steps_attempted": len(metrics_details["step_times_s"]),
                "last_step_completed": last_step_completed + 1,  # 1-based index
                "goal_achieved": goal_achieved,
                "final_step_success": final_step_success,  # Did any step fail critically?
                "avg_step_time_s": round(
                    sum(valid_step_times) / len(valid_step_times), 3
                )
                if valid_step_times
                else 0,
                "avg_perception_time_s": round(
                    sum(valid_perc_times) / len(valid_perc_times), 3
                )
                if valid_perc_times
                else 0,
                "avg_planning_time_s": round(
                    sum(valid_plan_times) / len(valid_plan_times), 3
                )
                if valid_plan_times
                else 0,
                "avg_execution_time_s": round(
                    sum(valid_exec_times) / len(valid_exec_times), 3
                )
                if valid_exec_times
                else 0,
                "avg_elements_per_step": round(
                    sum(valid_elem_counts) / len(valid_elem_counts), 1
                )
                if valid_elem_counts
                else 0,
                "avg_active_tracks_per_step": round(
                    sum(valid_track_counts) / len(valid_track_counts), 1
                )
                if valid_track_counts
                else 0,
                "successful_actions": sum(
                    1 for r in metrics_details["action_results"] if r is True
                ),
                "failed_actions": sum(
                    1 for r in metrics_details["action_results"] if r is False
                ),
            }
            full_metrics_data = {"summary": summary_metrics, "details": metrics_details}
            with open(metrics_path, "w") as f:
                json.dump(full_metrics_data, f, indent=4)
            logger.info(f"Saved run metrics to {metrics_path}")
            logger.info(f"Metrics Summary: {json.dumps(summary_metrics)}")
        except Exception as metrics_e:
            logger.warning(
                f"Could not save or summarize metrics: {metrics_e}", exc_info=True
            )

        # Save Structured Log Data (JSON Lines format)
        log_protocol_path = os.path.join(run_output_dir, "run_log.jsonl")
        try:
            with open(log_protocol_path, "w") as f:
                for step_data_dict in self.run_log_data:
                    # Use default=str as a fallback for non-serializable types
                    f.write(json.dumps(step_data_dict, default=str) + "\n")
            logger.info(f"Saved structured run log to {log_protocol_path}")
        except Exception as log_protocol_e:
            logger.warning(
                f"Could not save structured run log: {log_protocol_e}", exc_info=True
            )
