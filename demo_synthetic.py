# demo_synthetic.py
"""
OmniMCP Demo: Synthetic Perception -> LLM Planner -> Synthetic Action Validation.
Generates UI images and simulates the loop without real screen interaction.
"""

import os
import time

from omnimcp.core import LLMActionPlan, plan_action_for_ui

# Import necessary components from the project
from omnimcp.synthetic_ui import (
    draw_highlight,  # Use the original draw_highlight from synthetic_ui
    generate_login_screen,
    simulate_action,
)
from omnimcp.types import UIElement
from omnimcp.utils import logger

# NOTE ON REFACTORING:
# The main loop structure in this script (run_synthetic_planner_demo) is similar
# to the core logic now encapsulated in `omnimcp.agent_executor.AgentExecutor`.
# In the future, this synthetic demo could be refactored to:
# 1. Create synthetic implementations of the PerceptionInterface and ExecutionInterface.
# 2. Instantiate AgentExecutor with these synthetic components.
# 3. Call `agent_executor.run(...)`.
# This would further consolidate the core loop logic and allow testing the
# AgentExecutor orchestration with controlled, synthetic inputs/outputs.
# For now, this script remains separate to demonstrate the synthetic setup
# independently.


# --- Configuration ---
OUTPUT_DIR = "demo_output_multistep"
SAVE_IMAGES = True
MAX_STEPS = 6


def run_synthetic_planner_demo():
    """Runs the multi-step OmniMCP demo using synthetic UI and LLM planning."""
    logger.info("--- Starting OmniMCP Multi-Step Synthetic Demo ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initial State & Goal
    logger.info("Generating initial login screen...")
    try:
        # Use save_path to ensure initial image is saved
        image, elements = generate_login_screen(
            save_path=os.path.join(OUTPUT_DIR, "step_0_state_initial.png")
        )
    except Exception as e:
        logger.error(f"Failed to generate initial screen: {e}", exc_info=True)
        return

    user_goal = "Log in using username 'testuser' and password 'password123'"
    logger.info(f"User Goal: '{user_goal}'")

    action_history: list[str] = []
    goal_achieved_flag = False
    last_step_completed = -1

    # --- Main Loop ---
    for step in range(MAX_STEPS):
        logger.info(f"\n--- Step {step + 1}/{MAX_STEPS} ---")
        step_img_prefix = f"step_{step + 1}"

        # Save/Show current state *before* planning/highlighting
        current_state_img_path = os.path.join(
            OUTPUT_DIR, f"{step_img_prefix}_state.png"
        )
        if SAVE_IMAGES:
            try:
                image.save(current_state_img_path)
                logger.info(f"Saved current state to {current_state_img_path}")
            except Exception as save_e:
                logger.warning(f"Could not save step state image: {save_e}")

        # 2. Plan Next Action
        logger.info("Planning action with LLM...")
        llm_plan: LLMActionPlan | None = None
        target_element: UIElement | None = None
        try:
            llm_plan, target_element = plan_action_for_ui(
                elements=elements,
                user_goal=user_goal,
                action_history=action_history,
                step=step,
            )

            logger.info(f"LLM Reasoning: {llm_plan.reasoning}")
            logger.info(
                f"LLM Proposed Action: {llm_plan.action} on Element ID: {llm_plan.element_id}"
            )
            if llm_plan.text_to_type:
                logger.info(f"Text to Type: '{llm_plan.text_to_type}'")
            if llm_plan.key_info:
                logger.info(f"Key Info: '{llm_plan.key_info}'")
            logger.info(f"LLM Goal Complete Assessment: {llm_plan.is_goal_complete}")

            # 3. Check for Goal Completion Flag
            if llm_plan.is_goal_complete:
                logger.info(
                    "LLM flag indicates goal should be complete after this action."
                )
                goal_achieved_flag = True

            # --- Updated Validation Check ---
            if (
                not goal_achieved_flag
                and llm_plan.action == "click"
                and not target_element
            ):
                logger.error(
                    f"LLM planned 'click' on invalid element ID ({llm_plan.element_id}). Stopping."
                )
                break

            # 4. Visualize Planned Action (uses synthetic_ui.draw_highlight)
            highlight_img_path = os.path.join(
                OUTPUT_DIR, f"{step_img_prefix}_highlight.png"
            )
            if target_element:
                try:
                    highlighted_image = draw_highlight(
                        image,
                        target_element,
                        plan=llm_plan,
                        color="lime",
                        width=4,
                    )
                    if SAVE_IMAGES:
                        highlighted_image.save(highlight_img_path)
                        logger.info(
                            f"Saved highlighted action with text to {highlight_img_path}"
                        )
                except Exception as draw_e:
                    logger.warning(f"Could not save highlight image: {draw_e}")
            else:
                # For non-element actions like press_key, still save an image showing the state
                # before the action, potentially adding text annotation later if needed.
                if SAVE_IMAGES:
                    try:
                        image.save(
                            highlight_img_path.replace(
                                "_highlight.png", "_state_before_no_highlight.png"
                            )
                        )
                        logger.info("No target element, saved pre-action state.")
                    except Exception as save_e:
                        logger.warning(
                            f"Could not save pre-action state image: {save_e}"
                        )

            # Record action for history *before* simulation changes state
            action_desc = f"Action: {llm_plan.action}"
            if llm_plan.text_to_type:
                action_desc += f" '{llm_plan.text_to_type}'"
            if llm_plan.key_info:
                action_desc += f" Key='{llm_plan.key_info}'"
            if target_element:
                action_desc += (
                    f" on Element ID {target_element.id} ('{target_element.content}')"
                )
            action_history.append(action_desc)
            logger.debug(f"Added to history: {action_desc}")

            # 5. Simulate Action -> Get New State
            logger.info("Simulating action...")
            username = next(
                (
                    el.content
                    for el in elements
                    if el.id == 0 and el.type == "text_field"
                ),
                "User",
            )

            new_image, new_elements = simulate_action(
                image, elements, llm_plan, username_for_login=username
            )

            state_changed = (
                (id(new_image) != id(image))
                or (len(elements) != len(new_elements))
                or any(
                    e1.to_dict() != e2.to_dict()
                    for e1, e2 in zip(elements, new_elements)
                )
            )

            image, elements = new_image, new_elements

            if state_changed:
                logger.info(
                    f"State updated for next step. New element count: {len(elements)}"
                )
            else:
                logger.warning(
                    "Simulation did not result in a detectable state change."
                )

            last_step_completed = step

            # 6. NOW check the flag to break *after* simulation
            if goal_achieved_flag:
                logger.success(
                    "Goal completion flag was set, ending loop after simulation."
                )
                break

            time.sleep(1)

        except Exception as e:
            logger.error(f"Error during step {step + 1}: {e}", exc_info=True)
            break

    # --- End of Loop ---
    logger.info("\n--- Multi-Step Synthetic Demo Finished ---")
    if goal_achieved_flag:
        logger.success("Overall goal marked as achieved by LLM during execution.")
    elif last_step_completed == MAX_STEPS - 1:
        logger.warning(
            f"Reached maximum steps ({MAX_STEPS}) without goal completion flag being set."
        )
    else:
        logger.error(
            f"Execution stopped prematurely after Step {last_step_completed + 1} (check logs)."
        )

    # Save final state
    final_state_img_path = os.path.join(OUTPUT_DIR, "final_state.png")
    if SAVE_IMAGES:
        try:
            image.save(final_state_img_path)
            logger.info(f"Saved final state to {final_state_img_path}")
        except Exception as save_e:
            logger.warning(f"Could not save final state image: {save_e}")


if __name__ == "__main__":
    # Optional: Add check for API key, though planning might work differently
    # depending on whether core.plan_action_for_ui *requires* the LLM call
    # or could potentially use non-LLM logic someday.
    # from omnimcp.config import config
    # if not config.ANTHROPIC_API_KEY:
    #    logger.warning("ANTHROPIC_API_KEY not found. LLM planning might fail.")
    run_synthetic_planner_demo()
