# omnimcp/agent_executor.py

import datetime
import os
import time
from collections.abc import Callable
from typing import Protocol

from PIL import Image

from omnimcp import config, setup_run_logging
from omnimcp.types import LLMActionPlan, UIElement
from omnimcp.utils import (
    denormalize_coordinates,
    draw_action_highlight,
    draw_bounding_boxes,
    get_scaling_factor,
    logger,
    take_screenshot,
)


class PerceptionInterface(Protocol):
    elements: list[UIElement]
    screen_dimensions: tuple[int, int] | None
    _last_screenshot: Image.Image | None

    def update(self) -> None: ...


class ExecutionInterface(Protocol):
    def click(self, x: int, y: int, click_type: str = "single") -> bool: ...
    def type_text(self, text: str) -> bool: ...
    def execute_key_string(self, key_info_str: str) -> bool: ...
    def scroll(self, dx: int, dy: int) -> bool: ...


PlannerCallable = Callable[
    [list[UIElement], str, list[str], int, str],
    tuple[LLMActionPlan, UIElement | None],
]
ImageProcessorCallable = Callable[..., Image.Image]


# --- Core Agent Executor ---


class AgentExecutor:
    """
    Orchestrates the perceive-plan-act loop for UI automation tasks.
    Refactored to use action handlers for clarity.
    """

    def __init__(
        self,
        perception: PerceptionInterface,
        planner: PlannerCallable,
        execution: ExecutionInterface,
        box_drawer: ImageProcessorCallable | None = draw_bounding_boxes,
        highlighter: ImageProcessorCallable | None = draw_action_highlight,
    ):
        self._perception = perception
        self._planner = planner
        self._execution = execution
        self._box_drawer = box_drawer
        self._highlighter = highlighter
        self.action_history: list[str] = []

        # Map action names to their handler methods
        self._action_handlers: dict[str, Callable[..., bool]] = {
            "click": self._execute_click,
            "type": self._execute_type,
            "press_key": self._execute_press_key,
            "scroll": self._execute_scroll,
        }
        logger.info("AgentExecutor initialized with action handlers.")

    # --- Private Action Handlers ---

    def _execute_click(
        self,
        plan: LLMActionPlan,
        target_element: UIElement | None,
        screen_dims: tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'click' action."""
        if not target_element:
            logger.error(
                f"Click action requires target element ID {plan.element_id}, but it's missing."
            )
            return False  # Should have been caught earlier, but safety check

        screen_w, screen_h = screen_dims
        # Denormalize to get PHYSICAL PIXEL coordinates for center
        abs_x, abs_y = denormalize_coordinates(
            target_element.bounds[0],
            target_element.bounds[1],
            screen_w,
            screen_h,
            target_element.bounds[2],
            target_element.bounds[3],
        )
        # Convert to LOGICAL points for execution component
        logical_x = int(abs_x / scaling_factor)
        logical_y = int(abs_y / scaling_factor)
        logger.debug(f"Executing click at logical coords: ({logical_x}, {logical_y})")
        return self._execution.click(logical_x, logical_y, click_type="single")

    def _execute_type(
        self,
        plan: LLMActionPlan,
        target_element: UIElement | None,
        screen_dims: tuple[int, int],
        scaling_factor: int,
    ) -> bool:
        """Handles the 'type' action."""
        if plan.text_to_type is None:
            logger.error("Action 'type' planned but text_to_type is null.")
            return False  # Should be caught by Pydantic validation

        if target_element:  # Click target element first if specified
            screen_w, screen_h = screen_dims
            abs_x, abs_y = denormalize_coordinates(
                target_element.bounds[0],
                target_element.bounds[1],
                screen_w,
                screen_h,
                target_element.bounds[2],
                target_element.bounds[3],
            )
            logical_x = int(abs_x / scaling_factor)
            logical_y = int(abs_y / scaling_factor)
            logger.debug(
                f"Clicking target element {target_element.id} at logical ({logical_x},{logical_y}) before typing..."
            )
            if not self._execution.click(logical_x, logical_y):
                logger.warning(
                    "Failed to click target before typing, attempting type anyway."
                )
            time.sleep(0.2)  # Pause after click

        logger.debug(f"Executing type: '{plan.text_to_type[:50]}...'")
        return self._execution.type_text(plan.text_to_type)

    def _execute_press_key(
        self,
        plan: LLMActionPlan,
        target_element: UIElement | None,  # Unused, but maintains handler signature
        screen_dims: tuple[int, int],  # Unused
        scaling_factor: int,  # Unused
    ) -> bool:
        """Handles the 'press_key' action."""
        if not plan.key_info:
            logger.error("Action 'press_key' planned but key_info is null.")
            return False  # Should be caught by Pydantic validation
        logger.debug(f"Executing press_key: '{plan.key_info}'")
        return self._execution.execute_key_string(plan.key_info)

    def _execute_scroll(
        self,
        plan: LLMActionPlan,
        target_element: UIElement | None,  # Unused
        screen_dims: tuple[int, int],  # Unused
        scaling_factor: int,  # Unused
    ) -> bool:
        """Handles the 'scroll' action."""
        # Basic scroll logic based on reasoning hint
        scroll_dir = plan.reasoning.lower()
        scroll_amount_steps = 3
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
                "Scroll planned but direction/amount unclear, skipping scroll."
            )
            return True  # No action needed counts as success

    # Comparison Note:
    # This `run` method implements an explicit, sequential perceive-plan-act loop.
    # Alternative agent architectures exist, such as:
    # - ReAct (Reasoning-Acting): Where the LLM explicitly decides between
    #   reasoning steps and action steps.
    # - Callback-driven: Where UI events or timers might trigger agent actions.
    # - More complex state machines or graph-based execution flows.
    # This simple sequential loop provides a clear baseline. Future work might explore
    # these alternatives for more complex or reactive tasks.

    def run(
        self, goal: str, max_steps: int = 10, output_base_dir: str | None = None
    ) -> bool:
        """
        Runs the main perceive-plan-act loop to achieve the goal.

        Args:
            goal: The natural language goal for the agent.
            max_steps: Maximum number of steps to attempt.
            output_base_dir: Base directory to save run artifacts (timestamped).
                            If None, uses config.RUN_OUTPUT_DIR.

        Returns:
            True if the goal was achieved, False otherwise (error or max steps reached).
        """

        # Use configured output dir if none provided
        if output_base_dir is None:
            output_base_dir = config.RUN_OUTPUT_DIR

        run_timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .astimezone()
            .strftime("%Y%m%d_%H%M%S")
        )
        run_output_dir = os.path.join(output_base_dir, run_timestamp)

        try:
            os.makedirs(run_output_dir, exist_ok=True)

            # Configure run-specific logging
            log_path = setup_run_logging(run_output_dir)

            logger.info(f"Starting agent run. Goal: '{goal}'")
            logger.info(f"Saving outputs to: {run_output_dir}")
            logger.info(f"Run log file: {log_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {run_output_dir}: {e}")
            return False

        self.action_history = []
        goal_achieved = False
        final_step_success = True
        last_step_completed = -1

        try:
            scaling_factor = get_scaling_factor()
            logger.info(f"Using display scaling factor: {scaling_factor}")
        except Exception as e:
            logger.error(f"Failed to get scaling factor: {e}. Assuming 1.")
            scaling_factor = 1

        # --- Main Loop ---
        for step in range(max_steps):
            logger.info(f"\n--- Step {step + 1}/{max_steps} ---")
            step_start_time = time.time()
            step_img_prefix = f"step_{step + 1}"
            current_image: Image.Image | None = None
            current_elements: list[UIElement] = []
            screen_dimensions: tuple[int, int] | None = None

            # 1. Perceive State
            try:
                logger.debug("Perceiving current screen state...")
                self._perception.update()
                current_elements = self._perception.elements or []
                current_image = self._perception._last_screenshot
                screen_dimensions = self._perception.screen_dimensions

                if not current_image or not screen_dimensions:
                    raise RuntimeError("Failed to get valid screenshot or dimensions.")
                logger.info(f"Perceived state with {len(current_elements)} elements.")

            except Exception as perceive_e:
                logger.error(f"Perception failed: {perceive_e}", exc_info=True)
                final_step_success = False
                break

            # 2. Save State Artifacts (Unchanged)
            raw_state_path = os.path.join(
                run_output_dir, f"{step_img_prefix}_state_raw.png"
            )
            try:
                current_image.save(raw_state_path)
                logger.debug(f"Saved raw state image to {raw_state_path}")
            except Exception as save_raw_e:
                logger.warning(f"Could not save raw state image: {save_raw_e}")

            if self._box_drawer:
                parsed_state_path = os.path.join(
                    run_output_dir, f"{step_img_prefix}_state_parsed.png"
                )
                try:
                    img_with_boxes = self._box_drawer(
                        current_image, current_elements, color="lime", show_ids=True
                    )
                    img_with_boxes.save(parsed_state_path)
                    logger.debug(
                        f"Saved parsed state visualization to {parsed_state_path}"
                    )
                except Exception as draw_boxes_e:
                    logger.warning(f"Could not save parsed state image: {draw_boxes_e}")

            # 3. Plan Action (Unchanged)
            llm_plan: LLMActionPlan | None = None
            target_element: UIElement | None = None
            try:
                logger.debug("Planning next action...")
                llm_plan, target_element = self._planner(
                    elements=current_elements,
                    user_goal=goal,
                    action_history=self.action_history,
                    step=step,  # 0-based index
                )
                # (Logging of plan details remains here)
                logger.info(f"LLM Reasoning: {llm_plan.reasoning}")
                logger.info(
                    f"LLM Plan: Action={llm_plan.action}, TargetID={llm_plan.element_id}, GoalComplete={llm_plan.is_goal_complete}"
                )
                if llm_plan.text_to_type:
                    logger.info(f"LLM Plan: Text='{llm_plan.text_to_type[:50]}...'")
                if llm_plan.key_info:
                    logger.info(f"LLM Plan: KeyInfo='{llm_plan.key_info}'")

            except Exception as plan_e:
                logger.error(f"Planning failed: {plan_e}", exc_info=True)
                final_step_success = False
                break

            # 4. Check Goal Completion (Before Action) (Unchanged)
            if llm_plan.is_goal_complete:
                logger.success("LLM determined the goal is achieved!")
                goal_achieved = True
                last_step_completed = step
                break

            # 5. Validate Action Requirements (Unchanged)
            if llm_plan.action == "click" and target_element is None:
                logger.error(
                    f"Action 'click' planned for element ID {llm_plan.element_id}, but element not found. Stopping."
                )
                final_step_success = False
                break

            # 6. Visualize Planned Action (Unchanged)
            if self._highlighter and current_image:
                highlight_img_path = os.path.join(
                    run_output_dir, f"{step_img_prefix}_action_highlight.png"
                )
                try:
                    highlighted_image = self._highlighter(
                        current_image,
                        element=target_element,
                        plan=llm_plan,
                        color="red",
                        width=3,
                    )
                    highlighted_image.save(highlight_img_path)
                    logger.debug(f"Saved action visualization to {highlight_img_path}")
                except Exception as draw_highlight_e:
                    logger.warning(
                        f"Could not save action visualization image: {draw_highlight_e}"
                    )

            # 7. Update Action History (Before Execution) (Unchanged)
            action_desc = f"Step {step + 1}: Planned {llm_plan.action}"
            if target_element:
                action_desc += (
                    f" on ID {target_element.id} ('{target_element.content[:30]}...')"
                )
            if llm_plan.text_to_type:
                action_desc += f" Text='{llm_plan.text_to_type[:20]}...'"
            if llm_plan.key_info:
                action_desc += f" Key='{llm_plan.key_info}'"
            self.action_history.append(action_desc)
            logger.debug(f"Added to history: {action_desc}")

            # 8. Execute Action (Refactored)
            logger.info(f"Executing action: {llm_plan.action}...")
            action_success = False
            try:
                handler = self._action_handlers.get(llm_plan.action)
                if handler:
                    # Pass necessary arguments to the handler
                    action_success = handler(
                        plan=llm_plan,
                        target_element=target_element,
                        screen_dims=screen_dimensions,
                        scaling_factor=scaling_factor,
                    )
                else:
                    logger.error(
                        f"Execution handler for action type '{llm_plan.action}' not found."
                    )
                    action_success = False

                # Check execution result
                if not action_success:
                    logger.error(f"Action '{llm_plan.action}' execution failed.")
                    final_step_success = False
                    break
                else:
                    logger.success("Action executed successfully.")

            except Exception as exec_e:
                logger.error(
                    f"Exception during action execution: {exec_e}", exc_info=True
                )
                final_step_success = False
                break

            # Mark step as fully completed (Unchanged)
            last_step_completed = step

            # Wait for UI to settle (Unchanged)
            time.sleep(1.5)
            logger.debug(
                f"Step {step + 1} duration: {time.time() - step_start_time:.2f}s"
            )

        # --- End of Loop --- (Rest of the method remains the same)
        logger.info("\n--- Agent Run Finished ---")
        if goal_achieved:
            logger.success("Overall goal marked as achieved by LLM.")
        elif final_step_success and last_step_completed == max_steps - 1:
            logger.warning(
                f"Reached maximum steps ({max_steps}) without goal completion."
            )
        elif not final_step_success:
            logger.error(
                f"Execution stopped prematurely after Step {last_step_completed + 1} due to an error."
            )

        logger.info("Capturing final screen state...")
        final_state_img_path = os.path.join(run_output_dir, "final_state.png")
        try:
            final_image = take_screenshot()
            if final_image:
                final_image.save(final_state_img_path)
                logger.info(f"Saved final screen state to {final_state_img_path}")
            else:
                logger.warning("Could not capture final screenshot.")
        except Exception as save_final_e:
            logger.warning(f"Could not save final state image: {save_final_e}")

        logger.info(f"Run artifacts saved in: {run_output_dir}")
        return goal_achieved
