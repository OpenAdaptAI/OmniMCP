#!/usr/bin/env python
# cli.py

"""
Command-line interface for running OmniMCP agent tasks using AgentExecutor.
"""

import platform
import sys
import time

import fire

from omnimcp.utils import logger

# Default configuration
DEFAULT_OUTPUT_DIR = "runs"
DEFAULT_MAX_STEPS = 10
DEFAULT_GOAL = "Open calculator and compute 5 * 9"


def run(
    goal: str = DEFAULT_GOAL,
    max_steps: int = DEFAULT_MAX_STEPS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ci_mode: bool = False,
):
    """
    Runs the OmniMCP agent to achieve a specified goal.

    Args:
        goal: The natural language goal for the agent.
        max_steps: Maximum number of steps to attempt.
        output_dir: Base directory to save run artifacts (timestamped subdirs).
        ci_mode: Run in CI mode (skips API validation and actual execution).
    """
    # --- Initial Checks ---
    logger.info("--- OmniMCP CLI ---")

    # Skip import-time checks if we're in CI mode
    if ci_mode:
        logger.info("Running in CI mode - skipping credential checks and execution")
        return 0

    # Delay imports to avoid credential checks at import time
    try:
        # Import necessary components from the project
        from omnimcp.agent_executor import AgentExecutor
        from omnimcp.config import config
        from omnimcp.core import plan_action_for_ui
        from omnimcp.input import InputController, _pynput_error
        from omnimcp.omniparser.client import OmniParserClient
        from omnimcp.utils import (
            NSScreen,  # Check for AppKit on macOS
            draw_action_highlight,
            draw_bounding_boxes,
        )
        from omnimcp.visual_state import VisualState
    except ImportError as e:
        logger.critical(f"Required dependency not found: {e}")
        return 1

    logger.info("Performing initial checks...")
    success = True

    # 1. API Key Check
    if not config.ANTHROPIC_API_KEY:
        logger.critical(
            "❌ ANTHROPIC_API_KEY not found in config or .env file. LLM planning requires this."
        )
        success = False
    else:
        logger.info("✅ ANTHROPIC_API_KEY found.")

    # 2. pynput Check
    if _pynput_error:
        logger.critical(
            f"❌ Input control library (pynput) failed to load: {_pynput_error}"
        )
        logger.critical(
            "   Real action execution will not work. Is it installed and prerequisites met (e.g., display server)?"
        )
        success = False
    else:
        logger.info("✅ Input control library (pynput) loaded.")

    # 3. macOS Scaling Check
    if platform.system() == "darwin":
        if not NSScreen:
            logger.warning(
                "⚠️ AppKit (pyobjc-framework-Cocoa) not found or failed to import."
            )
            logger.warning(
                "   Coordinate scaling for Retina displays may be incorrect. Install with 'uv pip install pyobjc-framework-Cocoa'."
            )
        else:
            logger.info("✅ AppKit found for macOS scaling.")

    if not success:
        logger.error("Prerequisite checks failed. Exiting.")
        return 1

    # --- Component Initialization ---
    logger.info("\nInitializing components...")
    try:
        # OmniParser Client (handles deployment if URL not set)
        parser_client = OmniParserClient(
            server_url=config.OMNIPARSER_URL, auto_deploy=(not config.OMNIPARSER_URL)
        )
        logger.info(f"   - OmniParserClient ready (URL: {parser_client.server_url})")

        # Perception Component
        visual_state = VisualState(parser_client=parser_client)
        logger.info("   - VisualState (Perception) ready.")

        # Execution Component
        controller = InputController()
        logger.info("   - InputController (Execution) ready.")

        # Planner Function (already imported)
        logger.info("   - LLM Planner function ready.")

        # Visualization Functions (already imported)
        logger.info("   - Visualization functions ready.")

    except ImportError as e:
        logger.critical(
            f"❌ Component initialization failed due to missing dependency: {e}"
        )
        logger.critical(
            "   Ensure all requirements are installed (`uv pip install -e .`)"
        )
        return 1
    except Exception as e:
        logger.critical(f"❌ Component initialization failed: {e}", exc_info=True)
        return 1

    # --- Agent Executor Initialization ---
    logger.info("\nInitializing Agent Executor...")
    try:
        agent_executor = AgentExecutor(
            perception=visual_state,
            planner=plan_action_for_ui,
            execution=controller,
            box_drawer=draw_bounding_boxes,
            highlighter=draw_action_highlight,
        )
        logger.success("✅ Agent Executor initialized successfully.")
    except Exception as e:
        logger.critical(f"❌ Agent Executor initialization failed: {e}", exc_info=True)
        return 1

    # --- User Confirmation & Start ---
    print("\n" + "=" * 60)
    print(" WARNING: This script WILL take control of your mouse and keyboard!")
    print(f"          TARGET OS: {platform.system()}")
    print(" Please ensure no sensitive information is visible on screen.")
    print(" To stop execution manually: Move mouse RAPIDLY to a screen corner")
    print("                           OR press Ctrl+C in the terminal.")
    print("=" * 60 + "\n")
    for i in range(5, 0, -1):
        print(f"Starting in {i}...", end="\r")
        time.sleep(1)
    print("Starting agent run now!             ")

    # --- Run the Agent ---
    overall_success = False
    try:
        overall_success = agent_executor.run(
            goal=goal,
            max_steps=max_steps,
            output_base_dir=output_dir,
        )
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user (Ctrl+C).")
        return 1
    except Exception as run_e:
        logger.critical(
            f"\nAn unexpected error occurred during the agent run: {run_e}",
            exc_info=True,
        )
        return 1
    finally:
        # Optional: Add cleanup here if needed (e.g., stopping parser server)
        logger.info(
            "Reminder: If using auto-deploy, stop the parser server with "
            "'python -m omnimcp.omniparser.server stop' when finished."
        )

    # --- Exit ---
    if overall_success:
        logger.success("\nAgent run finished successfully (goal achieved).")
        return 0
    else:
        logger.error(
            "\nAgent run finished unsuccessfully (goal not achieved or error occurred)."
        )
        return 1


def main():
    """Main entry point that handles Fire's return code conversion."""
    result = fire.Fire(run)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
