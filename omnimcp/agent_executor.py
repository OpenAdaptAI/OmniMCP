# omnimcp/agent_executor.py

import datetime
import os
import time
from typing import Callable, List, Optional, Tuple, Protocol, Dict, Any
import json

from PIL import Image
from loguru import logger  # Use loguru

from omnimcp import config, setup_run_logging

# Import necessary types from omnimcp.types
from omnimcp.types import (
    LLMActionPlan,
    UIElement,
    ElementTrack,
    LoggedStep,
    ScreenAnalysis,
    ActionDecision,  # Placeholders for future use/logging
)

# SimpleElementTracker is used within VisualState, not directly here
# from omnimcp.tracking import SimpleElementTracker
from omnimcp.utils import (
    denormalize_coordinates,
    draw_action_highlight,
    draw_bounding_boxes,
    get_scaling_factor,
    take_screenshot,  # Keep for final screenshot
)

# --- Interface Definitions ---


class PerceptionInterface(Protocol):
    """Defines the expected interface for the perception component."""

    elements: List[UIElement]  # Current raw elements from parser
    tracked_elements_view: List[
        ElementTrack
    ]  # Current tracked elements view from tracker
    screen_dimensions: Optional[Tuple[int, int]]
    _last_screenshot: Optional[Image.Image]
    frame_counter: int  # The current frame/step number managed by perception

    def update(self) -> None: ...  # Updates all state including tracked_elements_view


class ExecutionInterface(Protocol):
    """Defines the expected interface for the execution component."""

    def click(self, x: int, y: int, click_type: str = "single") -> bool: ...
    def type_text(self, text: str) -> bool: ...
    def execute_key_string(self, key_info_str: str) -> bool: ...
    def scroll(self, dx: int, dy: int) -> bool: ...


# Updated PlannerCallable signature to accept tracking info
PlannerCallable = Callable[
    [  # Inputs:
        List[UIElement],  # Current raw elements for context
        str,  # User goal
        List[str],  # Action history descriptions
        int,  # Current step number
        Optional[List[ElementTrack]],  # Tracking info (list of current tracks)
    ],
    # Outputs:
    # Assume for now planner internally handles ActionDecision and converts back
    # to this tuple for compatibility with existing handlers.
    # This will change when core.py is fully reworked.
    Tuple[LLMActionPlan, Optional[UIElement]],
]

ImageProcessorCallable = Callable[..., Image.Image]

# --- Core Agent Executor ---


class AgentExecutor:
    """
    Orchestrates the perceive-plan-act loop, integrating perception with tracking,
    planning, execution, and structured logging.
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
        self._highlighter = highlighter
        self.action_history: List[str] = []
        self._action_handlers: Dict[str, Callable[..., bool]] = {
            "click": self._execute_click,
            "type": self._execute_type,
            "press_key": self._execute_press_key,
            "scroll": self._execute_scroll,
            # TODO: Add handlers for 'finish', 'wait' if added to action space
        }
        # Initialize metrics and structured log storage
        self.metrics: Dict[str, List[Any]] = self._reset_metrics()
        self.run_log_data: List[Dict] = []
        logger.info("AgentExecutor initialized.")

    def _reset_metrics(self) -> Dict[str, List[Any]]:
        """Helper to initialize/reset metrics dictionary for a run."""
        return {
            "step_times_s": [],
            "perception_times_s": [],
            "planning_times_s": [],
            "execution_times_s": [],
            "elements_per_step": [],
            "active_tracks_per_step": [],  # Added metric
            "action_results": [],  # Boolean success/fail
        }

    # --- Private Action Handlers ---
    # These currently consume LLMActionPlan. They might need updates
    # later if the planner starts returning ActionDecision directly to executor.

    def _execute_click(
        self,
        plan: LLMActionPlan,
        target_element: Optional[UIElement],
        screen_dims: Tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'click' action."""
        if not target_element:
            logger.error(
                f"Click planned for element ID {plan.element_id} but element not found by planner."
            )
            return False
        if not screen_dims:
            logger.error("Cannot execute click without screen dimensions.")
            return False

        # Denormalize using actual screen dimensions from perception
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
            f"Executing click at logical coords: ({logical_x}, {logical_y}) on Element ID {target_element.id}"
        )
        return self._execution.click(logical_x, logical_y, click_type="single")

    def _execute_type(
        self,
        plan: LLMActionPlan,
        target_element: Optional[UIElement],
        screen_dims: Tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'type' action."""
        if plan.text_to_type is None:
            logger.error("Action 'type' planned but text_to_type is null.")
            return False

        # Optional: Click target element first if specified
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
            time.sleep(0.2)  # Short pause after click

        logger.debug(f"Executing type: '{plan.text_to_type[:50]}...'")
        return self._execution.type_text(plan.text_to_type)

    def _execute_press_key(
        self,
        plan: LLMActionPlan,
        target_element: Optional[UIElement],
        screen_dims: Tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'press_key' action."""
        if not plan.key_info:
            logger.error("Action 'press_key' planned but key_info is null.")
            return False
        logger.debug(f"Executing press_key: '{plan.key_info}'")
        return self._execution.execute_key_string(plan.key_info)

    def _execute_scroll(
        self,
        plan: LLMActionPlan,
        target_element: Optional[UIElement],
        screen_dims: Tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'scroll' action."""
        # Basic scroll logic based on reasoning hint (can be improved)
        scroll_dir = plan.reasoning.lower()
        scroll_amount_steps = 3  # Arbitrary amount
        scroll_dy = (
            -scroll_amount_steps
            if "down" in scroll_dir
            else scroll_amount_steps
            if "up" in scroll_dir
            else 0
        )
        scroll_dx = (
            -scroll_amount_steps
            if "left" in scroll_dir
            else scroll_amount_steps
            if "right" in scroll_dir
            else 0
        )

        if scroll_dx != 0 or scroll_dy != 0:
            logger.debug(f"Executing scroll: dx={scroll_dx}, dy={scroll_dy}")
            return self._execution.scroll(scroll_dx, scroll_dy)
        else:
            logger.warning(
                "Scroll planned but direction unclear from reasoning, skipping scroll."
            )
            return True  # No action needed counts as success

    def run(
        self, goal: str, max_steps: int = 10, output_base_dir: Optional[str] = None
    ) -> bool:
        """Runs the main perceive-plan-act loop to achieve the goal."""
        # --- Setup ---
        if output_base_dir is None:
            output_base_dir = config.RUN_OUTPUT_DIR
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_base_dir, run_timestamp)
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            log_path = setup_run_logging(run_output_dir)
        except Exception as setup_e:
            logger.error(f"Failed during run setup (directory/logging): {setup_e}")
            return False
        logger.info(f"Starting agent run. Goal: '{goal}'")
        logger.info(f"Saving outputs to: {run_output_dir}")
        logger.info(f"Run log file: {log_path}")

        self.action_history = []
        self.metrics = self._reset_metrics()
        self.run_log_data = []
        goal_achieved = False
        final_step_success = True
        last_step_completed = -1
        # --- End Setup ---

        try:
            scaling_factor = get_scaling_factor()
            logger.info(f"Using display scaling factor: {scaling_factor}")
        except Exception as e:
            logger.error(f"Failed to get scaling factor: {e}. Assuming 1.")
            scaling_factor = 1

        # --- Main Loop ---
        for step in range(max_steps):
            step_start_time = time.time()
            logger.info(f"\n--- Step {step + 1}/{max_steps} ---")
            step_img_prefix = f"step_{step + 1}"

            # --- Initialize Step Variables ---
            current_image: Optional[Image.Image] = None
            current_elements: List[UIElement] = []
            tracked_elements_view: List[ElementTrack] = []
            screen_dimensions: Optional[Tuple[int, int]] = None
            tracking_info_for_log: Optional[List[Dict]] = None
            perception_duration = 0.0
            llm_plan: Optional[LLMActionPlan] = None  # Assumed output for now
            llm_analysis_log: Optional[Dict] = None  # Placeholder
            llm_decision_log: Optional[Dict] = None  # Placeholder
            target_element: Optional[UIElement] = None
            planning_duration = 0.0
            action_success = False
            executed_action_type = "none"
            executed_params: Dict[str, Any] = {}
            executed_target_id: Optional[int] = None
            execution_duration = 0.0
            step_screenshot_path: Optional[str] = None
            # --- End Initialize Step Variables ---

            # 1. Perceive State (including Tracking)
            perception_start_time = time.time()
            try:
                logger.debug("Updating visual state and tracking...")
                self._perception.update()  # This now internally calls the tracker

                # Retrieve results from the perception interface
                current_elements = self._perception.elements
                tracked_elements_view = self._perception.tracked_elements_view
                current_image = self._perception._last_screenshot
                screen_dimensions = self._perception.screen_dimensions
                perception_duration = time.time() - perception_start_time

                if not current_image or not screen_dimensions:
                    raise RuntimeError(
                        "Failed to get valid screenshot or dimensions during perception."
                    )

                logger.info(
                    f"Perceived state: {len(current_elements)} raw elements, "
                    f"{len(tracked_elements_view)} active tracks. "
                    f"Time: {perception_duration:.2f}s."
                )
                # Prepare tracking info for structured logging
                tracking_info_for_log = [
                    t.model_dump(mode="json") for t in tracked_elements_view
                ]

            except Exception as perceive_e:
                logger.error(f"Perception failed: {perceive_e}", exc_info=True)
                final_step_success = False
                # Log partial metrics
                self.metrics["perception_times_s"].append(
                    round(time.time() - perception_start_time, 3)
                )
                self.metrics["elements_per_step"].append(0)
                self.metrics["active_tracks_per_step"].append(0)
                # Attempt to log step failure before breaking
                step_duration = time.time() - step_start_time
                self._log_step_data(
                    step,
                    goal,
                    step_screenshot_path,
                    current_elements,
                    tracking_info_for_log,
                    None,
                    None,
                    None,
                    "perception_error",
                    None,
                    {},
                    False,
                    perception_duration,
                    0.0,
                    0.0,
                    step_duration,
                )
                break

            # Log perception metrics on success
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
                    logger.debug(f"Saved raw state image to {raw_state_path}")
                    if self._box_drawer:
                        # Draw boxes on raw elements for current frame visualization
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
                # Pass the tracked elements view to the planner
                llm_plan, target_element = self._planner(
                    elements=current_elements,  # Raw elements for context
                    user_goal=goal,
                    action_history=self.action_history,
                    step=step,
                    tracking_info=tracked_elements_view,  # Pass tracked view
                )
                planning_duration = time.time() - planning_start_time
                logger.info(f"Planning completed in {planning_duration:.2f}s.")

                if llm_plan:
                    # Log details from the plan
                    logger.info(f"LLM Reasoning: {llm_plan.reasoning}")
                    logger.info(
                        f"LLM Plan: Action={llm_plan.action}, TargetID={llm_plan.element_id}, "
                        f"GoalComplete={llm_plan.is_goal_complete}"
                    )
                    # Set execution details based on plan
                    executed_action_type = llm_plan.action
                    executed_target_id = llm_plan.element_id
                    executed_params = {}
                    if llm_plan.text_to_type is not None:
                        executed_params["text_to_type"] = llm_plan.text_to_type
                    if llm_plan.key_info is not None:
                        executed_params["key_info"] = llm_plan.key_info
                else:
                    raise ValueError("Planner returned None for LLMActionPlan")

            except Exception as plan_e:
                logger.error(f"Planning failed: {plan_e}", exc_info=True)
                final_step_success = False
                self.metrics["planning_times_s"].append(
                    round(time.time() - planning_start_time, 3)
                )
                step_duration = time.time() - step_start_time
                self._log_step_data(
                    step,
                    goal,
                    step_screenshot_path,
                    current_elements,
                    tracking_info_for_log,
                    None,
                    None,
                    None,
                    "planning_error",
                    None,
                    {},
                    False,
                    perception_duration,
                    planning_duration,
                    0.0,
                    step_duration,
                )
                break

            self.metrics["planning_times_s"].append(round(planning_duration, 3))

            # 4. Check Goal Completion
            if llm_plan.is_goal_complete:
                logger.success("LLM determined the goal is achieved!")
                goal_achieved = True
                # Log step data before potential break

            # 5. Validate Action Requirements
            if (
                llm_plan.action == "click"
                and target_element is None
                and not goal_achieved
            ):
                logger.error(
                    f"Action 'click' planned for element ID {llm_plan.element_id}, but element not found. Stopping."
                )
                final_step_success = False
                # Log step data before potential break

            # 6. Visualize Planned Action
            if self._highlighter and current_image and llm_plan:
                highlight_img_path = os.path.join(
                    run_output_dir, f"{step_img_prefix}_action_highlight.png"
                )
                try:
                    # Target element might be None if action doesn't require it
                    highlighted_image = self._highlighter(
                        current_image,
                        element=target_element,
                        plan=llm_plan,
                        color="red",
                        width=3,
                    )
                    highlighted_image.save(highlight_img_path)
                except Exception as draw_highlight_e:
                    logger.warning(
                        f"Could not save action visualization image: {draw_highlight_e}"
                    )

            # 7. Update Action History (Append before execution)
            action_desc = f"Step {step + 1}: Planned {llm_plan.action}"
            if target_element:
                action_desc += f" on ElemID {target_element.id}"
            if "text_to_type" in executed_params:
                action_desc += f" Text='{executed_params['text_to_type'][:20]}...'"
            if "key_info" in executed_params:
                action_desc += f" Key='{executed_params['key_info']}'"
            self.action_history.append(action_desc)
            logger.debug(f"Added to history: {action_desc}")

            # 8. Execute Action
            execution_start_time = time.time()
            if (
                not goal_achieved and final_step_success
            ):  # Only execute if needed and possible
                logger.info(f"Executing action: {executed_action_type}...")
                try:
                    handler = self._action_handlers.get(executed_action_type)
                    if handler:
                        action_success = handler(
                            plan=llm_plan,
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
                        final_step_success = False
                    else:
                        logger.success("Action executed successfully.")

                except Exception as exec_e:
                    logger.error(
                        f"Exception during action execution: {exec_e}", exc_info=True
                    )
                    action_success = False
                    final_step_success = False
            else:
                # Goal already met or prior failure, skip execution
                action_success = True  # Treat skipped step as 'successful' non-action
                logger.info(f"Skipping execution for step {step + 1}.")

            execution_duration = time.time() - execution_start_time

            # --- Log Execution Metrics and Action Result ---
            self.metrics["execution_times_s"].append(round(execution_duration, 3))
            self.metrics["action_results"].append(action_success)

            # --- Log Step Data to Protocol ---
            step_duration = time.time() - step_start_time
            self.metrics["step_times_s"].append(round(step_duration, 3))
            self._log_step_data(
                step,
                goal,
                step_screenshot_path,
                current_elements,
                tracking_info_for_log,
                llm_analysis_log,
                llm_decision_log,
                llm_plan,
                executed_action_type,
                executed_target_id,
                executed_params,
                action_success,
                perception_duration,
                planning_duration,
                execution_duration,
                step_duration,
            )

            # Check if run should terminate based on this step's outcome
            if goal_achieved or not final_step_success:
                last_step_completed = step
                break

            # Mark step completed if loop continues
            last_step_completed = step

            # Wait for UI to settle
            time.sleep(1.0)  # Make configurable or dynamic later

        # --- End of Loop ---
        logger.info("\n--- Agent Run Finished ---")
        if goal_achieved:
            logger.success("Overall goal marked as achieved.")
        elif not final_step_success:
            logger.error(f"Run failed at Step {last_step_completed + 1}.")
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

    def _log_step_data(
        self,
        step_index,
        goal,
        screenshot_path,
        elements,
        tracking_context,
        analysis,
        decision,
        raw_plan,
        exec_action,
        exec_target_id,
        exec_params,
        success,
        perc_time,
        plan_time,
        exec_time,
        step_time,
    ):
        """Helper to create and store the structured log entry for a step."""
        try:
            # Convert Pydantic models to dicts for logging if they are not None
            analysis_dict = (
                analysis.model_dump(mode="json")
                if isinstance(analysis, (ScreenAnalysis))
                else analysis
            )
            decision_dict = (
                decision.model_dump(mode="json")
                if isinstance(decision, (ActionDecision))
                else decision
            )
            raw_plan_dict = (
                raw_plan.model_dump(mode="json")
                if isinstance(raw_plan, (LLMActionPlan))
                else raw_plan
            )

            step_log_entry = LoggedStep(
                step_index=step_index,
                goal=goal,
                screenshot_path=screenshot_path,
                input_elements_count=len(elements),
                tracking_context=tracking_context,  # Assumes already list of dicts
                action_history_at_step=list(self.action_history),  # Use current history
                llm_analysis=analysis_dict,
                llm_decision=decision_dict,
                raw_llm_action_plan=raw_plan_dict,
                executed_action=exec_action,
                executed_target_element_id=exec_target_id,
                executed_parameters=exec_params,
                action_success=success,
                perception_time_s=round(perc_time, 3),
                planning_time_s=round(plan_time, 3),
                execution_time_s=round(exec_time, 3),
                step_time_s=round(step_time, 3),
            )
            self.run_log_data.append(step_log_entry.model_dump(mode="json"))
        except Exception as log_err:
            logger.warning(
                f"Failed to create or store structured log for step {step_index + 1}: {log_err}"
            )

    def _save_run_outputs(
        self, run_output_dir, goal_achieved, final_step_success, last_step_completed
    ):
        """Helper to save metrics and structured log data at the end of a run."""
        # Save Metrics
        metrics_path = os.path.join(run_output_dir, "run_metrics.json")
        try:
            # Calculate summary stats (handle empty lists)
            valid_step_times = [
                t for t in self.metrics["step_times_s"] if isinstance(t, (int, float))
            ]
            valid_perc_times = [
                t
                for t in self.metrics["perception_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_plan_times = [
                t
                for t in self.metrics["planning_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_exec_times = [
                t
                for t in self.metrics["execution_times_s"]
                if isinstance(t, (int, float))
            ]
            valid_elem_counts = [
                c for c in self.metrics["elements_per_step"] if isinstance(c, int)
            ]
            valid_track_counts = [
                c for c in self.metrics["active_tracks_per_step"] if isinstance(c, int)
            ]

            summary_metrics = {
                "total_steps_attempted": len(self.metrics["step_times_s"]),
                "last_step_completed": last_step_completed + 1,
                "goal_achieved": goal_achieved,
                "final_step_success": final_step_success,
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
                    1 for r in self.metrics["action_results"] if r is True
                ),
                "failed_actions": sum(
                    1 for r in self.metrics["action_results"] if r is False
                ),
            }
            full_metrics_data = {"summary": summary_metrics, "details": self.metrics}
            with open(metrics_path, "w") as f:
                json.dump(full_metrics_data, f, indent=4)
            logger.info(f"Saved run metrics to {metrics_path}")
            logger.info(f"Metrics Summary: {summary_metrics}")
        except Exception as metrics_e:
            logger.warning(f"Could not save or summarize metrics: {metrics_e}")

        # Save Structured Log Data
        log_protocol_path = os.path.join(run_output_dir, "run_log.jsonl")
        try:
            with open(log_protocol_path, "w") as f:
                for step_data_dict in self.run_log_data:
                    # Ensure complex objects within are serializable; model_dump helps
                    f.write(
                        json.dumps(step_data_dict, default=str) + "\n"
                    )  # Use default=str as fallback
            logger.info(f"Saved structured run log to {log_protocol_path}")
        except Exception as log_protocol_e:
            logger.warning(f"Could not save structured run log: {log_protocol_e}")
