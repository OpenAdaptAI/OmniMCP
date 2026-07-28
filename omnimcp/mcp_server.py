# omnimcp/mcp_server.py

import sys
import time
from typing import Literal

import numpy as np
from loguru import logger

# Use FastMCP from the official mcp package
from mcp.server.fastmcp import FastMCP
from PIL import Image

# Imports needed by OmniMCP class and its tools
from omnimcp.config import config  # Import config to read URL
from omnimcp.input import InputController

# Import parser client as it's needed to init VisualState here
from omnimcp.omniparser.client import OmniParserClient
from omnimcp.types import (
    ActionVerification,
    Bounds,
    InteractionResult,
    ScreenState,
    ScrollResult,
    TypeResult,
    UIElement,
)
from omnimcp.utils import compute_diff, denormalize_coordinates

# Import VisualState from its new location
from omnimcp.visual_state import VisualState


class OmniMCP:
    """
    Helper class to configure an MCP server for UI interaction.

    NOTE: This server implementation is experimental. It requires the
    OmniParser service to be running independently and its URL to be
    configured via the OMNIPARSER_URL variable in the project's .env file.

    To ensure the OmniParser service is running (if using auto-deploy):
    1. Check status: `python -m omnimcp.omniparser.server status`
    2. If stopped, start it: `python -m omnimcp.omniparser.server start`
       (Alternatively, running `python cli.py` might also start it).
    3. Note the URL provided (e.g., http://<ip_address>:8000).
    4. Add/uncomment the following line in your `.env` file:
       OMNIPARSER_URL=http://<ip_address>:8000
    """

    def __init__(self, debug: bool = False):
        """Initializes components and configures MCP tools."""
        logger.info(f"Initializing OmniMCP Server Components. Debug={debug}")

        parser_url_from_config = config.OMNIPARSER_URL
        if not parser_url_from_config:
            logger.critical(
                "MCP Server requires OMNIPARSER_URL to be set in config/env."
            )
            raise RuntimeError("MCP Server requires a pre-configured OMNIPARSER_URL.")
        logger.info(
            f"MCP Server using configured OmniParser URL: {parser_url_from_config}"
        )

        try:
            self._parser_client = OmniParserClient(
                server_url=parser_url_from_config,
                auto_deploy=False,  # Explicitly disable auto-deploy for server
            )
            logger.success("OmniParserClient configured successfully for MCP Server.")
        except Exception as client_init_e:
            logger.critical(
                f"MCP Server: Failed to configure or connect OmniParserClient using URL {parser_url_from_config}: {client_init_e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"OmniMCP Server failed to init OmniParserClient with URL {parser_url_from_config}"
            ) from client_init_e

        try:
            self._controller = InputController()
            logger.info("MCP Server: InputController initialized.")
        except ImportError as e:
            logger.critical(
                f"MCP Server: Failed to initialize InputController: {e}. Is pynput installed?"
            )
            raise RuntimeError(
                "OmniMCP Server cannot start without InputController"
            ) from e
        except Exception as controller_init_e:
            logger.critical(
                f"MCP Server: Failed to initialize InputController: {controller_init_e}",
                exc_info=True,
            )
            raise RuntimeError(
                "OmniMCP Server cannot start without InputController"
            ) from controller_init_e

        self._visual_state = VisualState(parser_client=self._parser_client)
        self._debug = debug

        self.mcp = FastMCP("omnimcp_server")
        self._setup_tools()
        logger.info("OmniMCP Server tools registered.")

    def _setup_tools(self):
        """Register MCP tools for UI interaction."""

        @self.mcp.tool()
        def get_screen_state() -> ScreenState:
            """Get current state of visible UI elements."""
            logger.info("MCP Tool: get_screen_state called")
            self._visual_state.update()
            return ScreenState(
                elements=self._visual_state.elements,
                dimensions=self._visual_state.screen_dimensions or (0, 0),
                timestamp=self._visual_state.timestamp or time.time(),
            )

        @self.mcp.tool()
        def describe_element(description: str) -> str:
            """Get rich description of UI element (Basic implementation)."""
            logger.info(f"MCP Tool: describe_element '{description}'")
            self._visual_state.update()
            element = self._visual_state.find_element(description)
            if not element:
                return f"No element found matching: {description}"
            # TODO: Enhance description with more detail or LLM integration.
            return f"Found {element.type} with content '{element.content}' at bounds {element.bounds}"

        @self.mcp.tool()
        def find_elements(query: str, max_results: int = 5) -> list[UIElement]:
            """Find elements matching natural query (Basic implementation)."""
            logger.info(f"MCP Tool: find_elements '{query}' (max: {max_results})")
            self._visual_state.update()
            # TODO: Enhance matching logic (e.g., vector search, LLM).
            matching_elements = []
            for element in self._visual_state.elements:
                if (
                    element.content
                    and any(
                        word in element.content.lower()
                        for word in query.lower().split()
                        if word
                    )
                    or (
                        element.type
                        and any(
                            word in element.type.lower()
                            for word in query.lower().split()
                            if word
                        )
                        and element not in matching_elements
                    )
                ):
                    matching_elements.append(element)
                if len(matching_elements) >= max_results:
                    break
            logger.info(
                f"MCP Tool: Found {len(matching_elements)} elements matching query."
            )
            return matching_elements

        @self.mcp.tool()
        def click_element(
            description: str,
            click_type: Literal["single", "double", "right"] = "single",
        ) -> InteractionResult:
            """Click UI element matching description. Returns immediately after action attempt."""
            logger.info(f"MCP Tool: click_element '{description}' (type: {click_type})")
            self._visual_state.update()
            element = self._visual_state.find_element(description)
            if not element:
                logger.error(f"MCP Tool: Element not found for click: {description}")
                return InteractionResult(
                    success=False,
                    element=None,
                    error=f"Element not found: {description}",
                )
            # Note: before_screenshot removed as verification is removed from this step
            logger.info(
                f"MCP Tool: Attempting {click_type} click on element ID {element.id}"
            )
            success, error_msg = False, None
            try:
                if self._visual_state.screen_dimensions:
                    w, h = self._visual_state.screen_dimensions
                    abs_x, abs_y = denormalize_coordinates(
                        element.bounds[0],
                        element.bounds[1],
                        w,
                        h,
                        element.bounds[2],
                        element.bounds[3],
                    )
                    logical_x, logical_y = abs_x, abs_y  # Assuming scale=1
                    logger.debug(
                        f"MCP Tool: Clicking at calculated coords ({logical_x}, {logical_y})"
                    )
                    success = self._controller.click(
                        logical_x, logical_y, click_type=click_type
                    )
                    if not success:
                        error_msg = (
                            f"InputController failed to perform {click_type} click."
                        )
                else:
                    error_msg, success = "Screen dimensions unknown.", False
            except Exception as click_e:
                logger.error(f"MCP Tool: Click action failed: {click_e}", exc_info=True)
                success, error_msg = False, f"Exception during click: {click_e}"
            # Note: verification=None in return
            return InteractionResult(
                success=success,
                element=element,
                verification=None,
                error=error_msg if not success else None,
            )

        @self.mcp.tool()
        def scroll_view(
            direction: Literal["up", "down", "left", "right"], amount: int = 1
        ) -> ScrollResult:
            """Scroll view in the specified direction. Returns immediately after action attempt."""
            logger.info(f"MCP Tool: scroll_view '{direction}' (amount: {amount})")
            scroll_steps, dx, dy = amount * 2, 0, 0
            if direction == "up":
                dy = scroll_steps
            elif direction == "down":
                dy = -scroll_steps
            elif direction == "left":
                dx = -scroll_steps
            elif direction == "right":
                dx = scroll_steps
            success, error_msg = True, None
            if dx != 0 or dy != 0:
                try:
                    success = self._controller.scroll(dx, dy)
                    if not success:
                        error_msg = "InputController failed to scroll."
                except Exception as scroll_e:
                    logger.error(
                        f"MCP Tool: Scroll action failed: {scroll_e}", exc_info=True
                    )
                    success, error_msg = False, f"Exception during scroll: {scroll_e}"
            else:
                logger.warning(
                    "MCP Tool: Scroll direction resulted in zero delta, skipping scroll."
                )
            # Note: verification=None in return
            return ScrollResult(
                success=success,
                element=None,
                scroll_amount=float(amount),
                verification=None,
                error=error_msg if not success else None,
            )

        @self.mcp.tool()
        def type_text(text: str, target: str | None = None) -> TypeResult:
            """
            Type text. If target description is provided, updates state, finds/clicks
            the target first. Otherwise, types immediately assuming focus is correct.
            Returns immediately after action attempt.
            """
            logger.info(f"MCP Tool: type_text '{text[:20]}...' (target: {target})")
            element = None

            # Only update state and click if a target is specified
            if target:
                logger.debug("Target specified, updating state and clicking...")
                self._visual_state.update()  # Update state to find the target
                element = self._visual_state.find_element(target)  # Find the target
                if not element:
                    logger.error(f"MCP Tool: Target element '{target}' not found.")
                    return TypeResult(
                        success=False,
                        element=None,
                        error=f"Target element not found: {target}",
                        text_entered="",
                    )

                # Click the found element
                logger.info(
                    f"MCP Tool: Clicking target element {element.id} before typing..."
                )
                click_success = False
                click_error_msg = None
                try:
                    if self._visual_state.screen_dimensions:
                        w, h = self._visual_state.screen_dimensions
                        abs_x, abs_y = denormalize_coordinates(
                            element.bounds[0],
                            element.bounds[1],
                            w,
                            h,
                            element.bounds[2],
                            element.bounds[3],
                        )
                        logical_x, logical_y = abs_x, abs_y  # Assuming scale=1
                        click_success = self._controller.click(
                            logical_x, logical_y, click_type="single"
                        )
                        if not click_success:
                            click_error_msg = "InputController failed click."
                    else:
                        click_error_msg, click_success = (
                            "Screen dimensions unknown.",
                            False,
                        )
                except Exception as click_e:
                    logger.error(
                        f"MCP Tool: Click on target failed: {click_e}", exc_info=True
                    )
                    click_success, click_error_msg = (
                        False,
                        f"Exception during click: {click_e}",
                    )

                if not click_success:
                    # Fail the whole operation if clicking the target fails
                    return TypeResult(
                        success=False,
                        element=element,
                        error=f"Failed to click target '{target}': {click_error_msg}",
                        text_entered="",
                    )
                time.sleep(0.2)  # Keep brief pause after successful click for focus
            else:
                # No target specified, proceed directly to typing
                logger.debug("No target specified, attempting to type directly.")

            # Attempt to type
            logger.info(f"MCP Tool: Attempting to type text: '{text[:20]}...'")
            success, error_msg = False, None
            try:
                success = self._controller.type_text(text)
                if not success:
                    error_msg = "InputController failed to type text."
            except Exception as type_e:
                logger.error(f"MCP Tool: Typing action failed: {type_e}", exc_info=True)
                success, error_msg = False, f"Exception during typing: {type_e}"

            # Return result (no second update or verification)
            return TypeResult(
                success=success,
                element=element,  # Target element if clicked, else None
                text_entered=text if success else "",
                verification=None,
                error=error_msg if not success else None,
            )

        @self.mcp.tool()
        def press_key(key_info: str) -> InteractionResult:
            """Press a key or key combination. Returns immediately after action attempt."""
            logger.info(f"MCP Tool: press_key '{key_info}'")
            # Note: before_screenshot removed as verification is removed from this step
            success, error_msg = False, None
            try:
                success = self._controller.execute_key_string(key_info)
                if not success:
                    error_msg = (
                        f"InputController failed to execute key string: {key_info}"
                    )
            except Exception as press_e:
                logger.error(
                    f"MCP Tool: Key press action failed for '{key_info}': {press_e}",
                    exc_info=True,
                )
                success, error_msg = (
                    False,
                    f"Exception during key press for '{key_info}': {press_e}",
                )
            # Note: verification=None in return
            return InteractionResult(
                success=success,
                element=None,
                context={"key_info": key_info},
                verification=None,
                error=error_msg if not success else None,
            )

    # _verify_action is kept as a helper, though not called by default tools now
    def _verify_action(
        self,
        before_image: Image.Image | None,
        after_image: Image.Image | None,
        element_bounds: Bounds | None = None,
        action_description: str | None = None,
    ) -> ActionVerification | None:
        """Verify action success using basic pixel difference."""
        # TODO: Refactor verification logic: Consider moving to a dedicated verification module, improving the diff algorithm (e.g., structural diff), or making verification optional via config due to performance impact and current basic implementation.
        logger.debug("MCP Tool: Verifying action using pixel difference...")
        if not before_image or not after_image:
            logger.warning(
                "MCP Tool: Cannot verify action, missing before or after image."
            )
            return ActionVerification(
                success=False,
                confidence=0.0,
                changes_detected=[],
                before_state=None,
                after_state=None,
            )
        try:
            diff_image = compute_diff(before_image, after_image)
            diff_array = np.array(diff_image)
            change_threshold = 30
            min_changed_pixels = 50
            changes = 0
            total_pixels_in_roi = diff_array.size if diff_array.size > 0 else 1
            if element_bounds and self._visual_state.screen_dimensions:
                img_width, img_height = self._visual_state.screen_dimensions
                x0, y0, x1, y1 = (
                    max(0, int(element_bounds[0] * img_width)),
                    max(0, int(element_bounds[1] * img_height)),
                    min(
                        img_width,
                        int((element_bounds[0] + element_bounds[2]) * img_width),
                    ),
                    min(
                        img_height,
                        int((element_bounds[1] + element_bounds[3]) * img_height),
                    ),
                )
                if x1 > x0 and y1 > y0:
                    roi = diff_array[y0:y1, x0:x1]
                    if roi.size > 0:
                        changes, total_pixels_in_roi = (
                            np.sum(roi > change_threshold),
                            roi.size,
                        )
                    else:
                        changes = 0
                else:
                    logger.warning(f"MCP Tool: Invalid bounds {element_bounds}...")
                    changes = np.sum(diff_array > change_threshold)
            else:
                changes = np.sum(diff_array > change_threshold)
            success = bool(changes > min_changed_pixels)
            confidence = (
                min(1.0, changes / max(1, total_pixels_in_roi * 0.001))
                if success
                else 0.0
            )
            logger.info(
                f"MCP Tool: Action verification: Changed pixels={changes}, Success={success}, Confidence={confidence:.2f}"
            )
            before_bytes, after_bytes = None, None
            return ActionVerification(
                success=success,
                before_state=before_bytes,
                after_state=after_bytes,
                changes_detected=[element_bounds] if element_bounds else [],
                confidence=float(confidence),
            )
        except Exception as e:
            logger.error(
                f"MCP Tool: Error during action verification: {e}", exc_info=True
            )
            return ActionVerification(
                success=False,
                confidence=0.0,
                changes_detected=[],
                before_state=None,
                after_state=None,
            )


# --- Module-Level Instantiation ---
try:
    omni_mcp_config = OmniMCP()
    mcp = omni_mcp_config.mcp
except Exception as e:
    logger.critical(f"Failed to initialize OmniMCP configuration: {e}", exc_info=True)
    mcp = None
    if __name__ == "__main__":
        sys.exit(1)

# --- Direct Execution Block ---
if __name__ == "__main__":
    if mcp:
        logger.info("Attempting to run OmniMCP Server directly using mcp.run()...")
        try:
            mcp.run()
        except KeyboardInterrupt:
            logger.info("OmniMCP Server stopping...")
        except Exception as main_e:
            logger.critical(
                f"Unexpected error running OmniMCP server: {main_e}", exc_info=True
            )
            sys.exit(1)
        logger.info("OmniMCP Server finished.")
    else:
        logger.error("MCP Server object ('mcp') could not be initialized. Cannot run.")
        sys.exit(1)
