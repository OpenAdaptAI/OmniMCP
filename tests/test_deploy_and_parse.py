# tests/test_deploy_and_parse.py

"""
A simple script to test OmniParser deployment, screenshotting,
parsing, and mapping to UIElements using VisualState.
"""

import asyncio  # Needed for async VisualState.update()
import sys

from omnimcp.omniparser.client import OmniParserClient
from omnimcp.utils import logger
from omnimcp.visual_state import VisualState

if __name__ == "__main__":
    logger.info("--- Starting OmniParser Integration Test ---")

    # 1. Initialize Client (Triggers auto-deploy/discovery)
    logger.info("Initializing OmniParserClient...")
    parser_client = None
    try:
        parser_client = OmniParserClient(auto_deploy=True)
        logger.success(
            f"OmniParserClient ready. Server URL: {parser_client.server_url}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize OmniParserClient: {e}", exc_info=True)
        sys.exit(1)

    # 2. Initialize VisualState
    logger.info("Initializing VisualState...")
    visual_state_manager = VisualState(parser_client=parser_client)

    # 3. Update Visual State (Takes screenshot, parses, maps)
    logger.info(
        "Updating visual state (this takes screenshot, calls parser, maps results)..."
    )
    try:
        # Run the async update function
        asyncio.run(visual_state_manager.update())

        if not visual_state_manager.elements:
            logger.warning("VisualState update completed, but no elements were mapped.")
            logger.warning(
                "Check OmniParser logs on the server or previous log messages for parser errors."
            )
        else:
            logger.success(
                f"VisualState update successful. Mapped {len(visual_state_manager.elements)} elements."
            )
            logger.info("First 5 mapped UI Elements:")
            for i, element in enumerate(visual_state_manager.elements[:5]):
                # Use a more readable format, perhaps to_prompt_repr or just key attributes
                print(
                    f"  {i}: ID={element.id}, Type={element.type}, Content='{element.content[:50]}...', Bounds={element.bounds}"
                )

            # You could now potentially pass visual_state_manager.elements to a planner
            # logger.info("Next step would be to call the planner with these elements.")

    except Exception as e:
        logger.error(f"Error during VisualState update: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Test Finished ---")
    logger.info(
        "Reminder: Run 'python omnimcp/omniparser/server.py stop' to shut down the EC2 instance."
    )
