# omnimcp/omnimcp.py

"""
OmniMCP: Model Context Protocol for UI Automation through visual understanding.
Refactored to use OmniParserClient.
"""

import time
from typing import List, Optional, Literal, Dict, Tuple

import numpy as np
from mcp.server.fastmcp import FastMCP
from loguru import logger
from PIL import Image

from omnimcp.omniparser.client import OmniParserClient

from omnimcp.utils import (
    take_screenshot,
    compute_diff,
    MouseController,
    KeyboardController,
)
from omnimcp.types import (
    Bounds,
    UIElement,
    ScreenState,
    ActionVerification,
    InteractionResult,
    ScrollResult,
    TypeResult,
)
# Assuming InputController uses Mouse/KeyboardController internally or replace its usage
# from omnimcp.input import InputController # Keep if exists and is used


class VisualState:
    """Manages the current state of visible UI elements."""

    # Modified __init__ to accept the client instance
    def __init__(self, parser_client: OmniParserClient):
        """Initialize the visual state manager.

        Args:
            parser_client: An initialized OmniParserClient instance.
        """
        self.elements: List[UIElement] = []
        self.timestamp: Optional[float] = None
        self.screen_dimensions: Optional[Tuple[int, int]] = None
        self._last_screenshot: Optional[Image.Image] = None
        # Store the passed-in client instance
        self._parser_client = parser_client
        if not self._parser_client:
            # This shouldn't happen if initialized correctly by OmniMCP
            logger.error("VisualState initialized without a valid parser_client!")
            raise ValueError("VisualState requires a valid OmniParserClient instance.")

    async def update(self):
        """Update visual state from screenshot using the parser client."""
        logger.debug("Updating VisualState...")
        try:
            # Capture screenshot
            screenshot = take_screenshot()
            self._last_screenshot = screenshot
            self.screen_dimensions = screenshot.size
            logger.debug(f"Screenshot taken: {self.screen_dimensions}")

            # Process with UI parser client
            # The client's __init__ should have already ensured the server is available/deployed
            if not self._parser_client or not self._parser_client.server_url:
                logger.error(
                    "OmniParser client or server URL not available for update."
                )
                # Decide behavior: return old state, raise error? Let's clear elements.
                self.elements = []
                self.timestamp = time.time()
                return self

            logger.debug(
                f"Parsing screenshot with client connected to {self._parser_client.server_url}"
            )
            # Call the parse_image method on the client instance
            parser_result = self._parser_client.parse_image(screenshot)

            # Update state based on results
            self._update_elements_from_parser(parser_result)
            self.timestamp = time.time()
            logger.debug(f"VisualState updated with {len(self.elements)} elements.")

        except Exception as e:
            logger.error(f"Failed to update visual state: {e}", exc_info=True)
            # Clear elements on error to indicate failure? Or keep stale data? Clear is safer.
            self.elements = []
            self.timestamp = time.time()  # Still update timestamp

        return self

    def _update_elements_from_parser(self, parser_result: Dict):
        """Process parser results dictionary into UIElements."""
        self.elements = []  # Start fresh

        if not isinstance(parser_result, dict):
            logger.error(f"Parser result is not a dictionary: {type(parser_result)}")
            return

        if "error" in parser_result:
            logger.error(f"Parser returned an error: {parser_result['error']}")
            return

        # Adjust key based on actual OmniParser output if different
        raw_elements = parser_result.get("parsed_content_list", [])
        if not isinstance(raw_elements, list):
            logger.error(
                f"Expected 'parsed_content_list' to be a list, got: {type(raw_elements)}"
            )
            return

        element_id_counter = 0
        for element_data in raw_elements:
            if not isinstance(element_data, dict):
                logger.warning(f"Skipping non-dict element data: {element_data}")
                continue
            # Pass screen dimensions for normalization
            ui_element = self._convert_to_ui_element(element_data, element_id_counter)
            if ui_element:
                self.elements.append(ui_element)
                element_id_counter += 1

    def _convert_to_ui_element(
        self, element_data: Dict, element_id: int
    ) -> Optional[UIElement]:
        """Convert parser element dict to UIElement dataclass."""
        try:
            # Extract and normalize bounds - requires screen_dimensions to be set
            if not self.screen_dimensions:
                logger.error("Cannot normalize bounds, screen dimensions not set.")
                return None
            # Assuming OmniParser returns relative [x_min, y_min, x_max, y_max]
            bbox_rel = element_data.get("bbox")
            if not isinstance(bbox_rel, list) or len(bbox_rel) != 4:
                logger.warning(f"Skipping element due to invalid bbox: {bbox_rel}")
                return None

            x_min_rel, y_min_rel, x_max_rel, y_max_rel = bbox_rel
            width_rel = x_max_rel - x_min_rel
            height_rel = y_max_rel - y_min_rel

            # Basic validation
            if not (
                0 <= x_min_rel <= 1
                and 0 <= y_min_rel <= 1
                and 0 <= width_rel <= 1
                and 0 <= height_rel <= 1
                and width_rel > 0
                and height_rel > 0
            ):
                logger.warning(
                    f"Skipping element due to invalid relative bbox values: {bbox_rel}"
                )
                return None

            bounds: Bounds = (x_min_rel, y_min_rel, width_rel, height_rel)

            # Map element type if needed (e.g., 'TextBox' -> 'text_field')
            element_type = (
                str(element_data.get("type", "unknown")).lower().replace(" ", "_")
            )

            # Create UIElement
            return UIElement(
                id=element_id,  # Assign sequential ID
                type=element_type,
                content=str(element_data.get("content", "")),
                bounds=bounds,
                confidence=float(element_data.get("confidence", 0.0)),  # Ensure float
                attributes=element_data.get("attributes", {}) or {},  # Ensure dict
            )
        except Exception as e:
            logger.error(
                f"Error converting element data {element_data}: {e}", exc_info=True
            )
            return None

    # find_element needs to be updated to use LLM or a better matching strategy
    def find_element(self, description: str) -> Optional[UIElement]:
        """Find UI element matching description (placeholder implementation)."""
        logger.debug(f"Finding element described as: '{description}'")
        if not self.elements:
            logger.warning("find_element called but no elements in current state.")
            return None

        # TODO: Replace this simple logic with LLM-based semantic search/matching
        # or a more robust fuzzy matching algorithm.
        search_terms = description.lower().split()
        best_match = None
        highest_score = 0

        for element in self.elements:
            content_lower = element.content.lower()
            type_lower = element.type.lower()
            score = 0
            for term in search_terms:
                # Give points for matching content or type
                if term in content_lower:
                    score += 2
                if term in type_lower:
                    score += 1
            # Basic proximity or relationship checks could be added here

            if score > highest_score:
                highest_score = score
                best_match = element
            elif score == highest_score and score > 0:
                # Handle ties? For now, just take the first best match.
                # Could prioritize interactive elements or larger elements?
                pass

        if best_match:
            logger.info(
                f"Found best match (score={highest_score}) for '{description}': ID={best_match.id}, Type={best_match.type}, Content='{best_match.content}'"
            )
        else:
            logger.warning(f"No element found matching description: '{description}'")

        return best_match


class OmniMCP:
    """Model Context Protocol server for UI understanding."""

    # Modified __init__ to accept/create OmniParserClient
    def __init__(self, parser_url: Optional[str] = None, debug: bool = False):
        """Initialize the OmniMCP server.

        Args:
            parser_url: Optional URL for an *existing* OmniParser service.
                        If None, a client with auto-deploy=True will be created.
            debug: Whether to enable debug mode (currently affects logging).
        """
        # Create the client here - it handles deployment/connection checks
        # Pass parser_url if provided, otherwise let client handle auto_deploy
        logger.info(f"Initializing OmniMCP. Debug={debug}")
        try:
            self._parser_client = OmniParserClient(
                server_url=parser_url, auto_deploy=(parser_url is None)
            )
            logger.success("OmniParserClient initialized within OmniMCP.")
        except Exception as client_init_e:
            logger.critical(
                f"Failed to initialize OmniParserClient needed by OmniMCP: {client_init_e}",
                exc_info=True,
            )
            # Depending on desired behavior, maybe raise or set a failed state
            raise RuntimeError(
                "OmniMCP cannot start without a working OmniParserClient"
            ) from client_init_e

        # Initialize other components, passing the client to VisualState
        # self.input = InputController() # Keep if used
        self.mcp = FastMCP("omnimcp")
        # Pass the initialized client to VisualState
        self._visual_state = VisualState(parser_client=self._parser_client)
        self._mouse = MouseController()  # Keep standard controllers
        self._keyboard = KeyboardController()
        self._debug = debug
        self._debug_context = None  # Keep for potential future debug features

        # Setup MCP tools after components are initialized
        self._setup_tools()
        logger.info("OmniMCP initialization complete. Tools registered.")

    def _setup_tools(self):
        """Register MCP tools"""

        # Decorator syntax seems slightly off for instance method, should use self.mcp.tool
        @self.mcp.tool()
        async def get_screen_state() -> ScreenState:
            """Get current state of visible UI elements"""
            logger.info("Tool: get_screen_state called")
            # Ensure visual state is updated before returning
            await self._visual_state.update()
            return ScreenState(
                elements=self._visual_state.elements,
                dimensions=self._visual_state.screen_dimensions
                or (0, 0),  # Handle None case
                timestamp=self._visual_state.timestamp or time.time(),
            )

        @self.mcp.tool()
        async def describe_element(description: str) -> str:
            """Get rich description of UI element"""
            logger.info(f"Tool: describe_element called with: '{description}'")
            # Update is needed to find based on latest screen
            await self._visual_state.update()
            element = self._visual_state.find_element(description)
            if not element:
                return f"No element found matching: {description}"
            # TODO: Enhance with LLM description generation later
            return (
                f"Found ID={element.id}: {element.type} with content '{element.content}' "
                f"at bounds {element.bounds}"
            )

        @self.mcp.tool()
        async def find_elements(query: str, max_results: int = 5) -> List[UIElement]:
            """Find elements matching natural query"""
            logger.info(
                f"Tool: find_elements called with query: '{query}', max_results={max_results}"
            )
            await self._visual_state.update()
            # Use the internal find_element logic which is currently basic matching
            # TODO: Implement better multi-element matching maybe using LLM embeddings later
            matching_elements = []
            for element in self._visual_state.elements:
                content_match = any(
                    word in element.content.lower() for word in query.lower().split()
                )
                type_match = any(
                    word in element.type.lower() for word in query.lower().split()
                )
                if content_match or type_match:
                    matching_elements.append(element)
                    if len(matching_elements) >= max_results:
                        break
            logger.info(f"Found {len(matching_elements)} elements for query.")
            return matching_elements

        @self.mcp.tool()
        async def click_element(
            description: str,
            click_type: Literal["single", "double", "right"] = "single",
        ) -> InteractionResult:
            """Click UI element matching description"""
            logger.info(f"Tool: click_element '{description}' (type: {click_type})")
            await self._visual_state.update()
            element = self._visual_state.find_element(description)
            if not element:
                logger.error(f"Element not found for click: {description}")
                return InteractionResult(
                    success=False,
                    element=None,
                    error=f"Element not found: {description}",
                )

            before_screenshot = self._visual_state._last_screenshot
            logger.info(f"Attempting {click_type} click on element ID {element.id}")
            # Use the simpler controllers directly for now
            # TODO: Integrate InputController if it adds value (e.g., smoother movement)
            try:
                # Convert bounds to absolute center
                if self._visual_state.screen_dimensions:
                    w, h = self._visual_state.screen_dimensions
                    abs_x = int((element.bounds[0] + element.bounds[2] / 2) * w)
                    abs_y = int((element.bounds[1] + element.bounds[3] / 2) * h)
                    self._mouse.move(abs_x, abs_y)
                    time.sleep(0.1)  # Short pause after move
                    if click_type == "single":
                        self._mouse.click(button="left")
                    elif click_type == "double":
                        self._mouse.double_click(
                            button="left"
                        )  # Assuming controller has double_click
                    elif click_type == "right":
                        self._mouse.click(button="right")
                    success = True
                    logger.success(
                        f"Performed {click_type} click at ({abs_x}, {abs_y})"
                    )
                else:
                    logger.error(
                        "Screen dimensions unknown, cannot calculate click coordinates."
                    )
                    success = False
            except Exception as click_e:
                logger.error(f"Click action failed: {click_e}", exc_info=True)
                success = False

            time.sleep(0.5)  # Wait for UI to potentially react
            await self._visual_state.update()  # Update state *after* action
            verification = await self._verify_action(
                before_screenshot, self._visual_state._last_screenshot, element.bounds
            )

            return InteractionResult(
                success=success,
                element=element,
                verification=verification,
                error="Click failed" if not success else None,
            )

        @self.mcp.tool()
        async def type_text(text: str, target: Optional[str] = None) -> TypeResult:
            """Type text, optionally clicking a target element first"""
            logger.info(f"Tool: type_text '{text}' (target: {target})")
            await self._visual_state.update()
            element = None
            # If target specified, try to click it
            if target:
                logger.info(f"Clicking target '{target}' before typing...")
                click_result = await click_element(
                    target, click_type="single"
                )  # Use the tool function
                if not click_result.success:
                    logger.error(
                        f"Failed to click target '{target}': {click_result.error}"
                    )
                    return TypeResult(
                        success=False,
                        element=None,
                        error=f"Failed to click target: {target}",
                        text_entered="",
                    )
                element = click_result.element
                time.sleep(0.2)  # Pause after click before typing

            before_screenshot = self._visual_state._last_screenshot
            logger.info(f"Attempting to type text: '{text}'")
            try:
                self._keyboard.type(text)
                success = True
                logger.success("Text typed.")
            except Exception as type_e:
                logger.error(f"Typing action failed: {type_e}", exc_info=True)
                success = False

            time.sleep(0.5)  # Wait for UI potentially
            await self._visual_state.update()
            verification = await self._verify_action(
                before_screenshot, self._visual_state._last_screenshot
            )

            return TypeResult(
                success=success,
                element=element,
                text_entered=text if success else "",
                verification=verification,
                error="Typing failed" if not success else None,
            )

        # Keep press_key and scroll_view as placeholders or implement fully
        @self.mcp.tool()
        async def press_key(key: str, modifiers: List[str] = None) -> InteractionResult:
            """Press keyboard key with optional modifiers"""
            logger.info(f"Tool: press_key '{key}' (modifiers: {modifiers})")
            # ... (update state, take screenshot, use self._keyboard.press, verify) ...
            logger.warning("press_key not fully implemented yet.")
            return InteractionResult(
                success=True,
                element=None,
                context={"key": key, "modifiers": modifiers or []},
            )

        @self.mcp.tool()
        async def scroll_view(
            direction: Literal["up", "down", "left", "right"], amount: int = 1
        ) -> ScrollResult:
            """Scroll the view in a specified direction by a number of units (e.g., mouse wheel clicks)."""
            logger.info(f"Tool: scroll_view {direction} {amount}")
            # ... (update state, take screenshot, use self._mouse.scroll, verify) ...
            logger.warning("scroll_view not fully implemented yet.")
            try:
                scroll_x = 0
                scroll_y = 0
                scroll_factor = amount  # Treat amount as wheel clicks/units
                if direction == "up":
                    scroll_y = scroll_factor
                elif direction == "down":
                    scroll_y = -scroll_factor
                elif direction == "left":
                    scroll_x = -scroll_factor
                elif direction == "right":
                    scroll_x = scroll_factor

                if scroll_x != 0 or scroll_y != 0:
                    self._mouse.scroll(scroll_x, scroll_y)
                    success = True
                else:
                    success = False  # No scroll happened

            except Exception as scroll_e:
                logger.error(f"Scroll action failed: {scroll_e}", exc_info=True)
                success = False

            # Add delay and state update/verification if needed
            time.sleep(0.5)
            # await self._visual_state.update() # Optional update after scroll
            # verification = ...

            return ScrollResult(
                success=success,
                scroll_amount=amount,
                direction=direction,
                verification=None,
            )  # Add verification later

    # Keep _verify_action, but note it relies on Claude or simple diff for now
    async def _verify_action(
        self, before_image, after_image, element_bounds=None, action_description=None
    ) -> Optional[ActionVerification]:
        """Verify action success (placeholder/basic diff)."""
        logger.debug("Verifying action...")
        if not before_image or not after_image:
            logger.warning("Cannot verify action, missing before or after image.")
            return None

        # Basic pixel diff verification (as implemented before)
        try:
            diff_image = compute_diff(before_image, after_image)
            diff_array = np.array(diff_image)
            # Consider only changes within bounds if provided
            change_threshold = 30  # Pixel value difference threshold
            min_changed_pixels = 50  # Minimum number of pixels changed significantly

            if element_bounds and self.screen_dimensions:
                w, h = self.screen_dimensions
                x0 = int(element_bounds[0] * w)
                y0 = int(element_bounds[1] * h)
                x1 = int((element_bounds[0] + element_bounds[2]) * w)
                y1 = int((element_bounds[1] + element_bounds[3]) * h)
                roi = diff_array[y0:y1, x0:x1]
                changes = np.sum(roi > change_threshold) if roi.size > 0 else 0
                total_pixels = roi.size if roi.size > 0 else 1
            else:
                changes = np.sum(diff_array > change_threshold)
                total_pixels = diff_array.size if diff_array.size > 0 else 1

            success = changes > min_changed_pixels
            confidence = (
                min(1.0, changes / max(1, total_pixels * 0.001)) if success else 0.0
            )  # Simple confidence metric
            logger.info(
                f"Action verification: Changed pixels={changes}, Success={success}, Confidence={confidence:.2f}"
            )

            # Store images as bytes (optional, can be large)
            # before_bytes_io = io.BytesIO(); before_image.save(before_bytes_io, format="PNG")
            # after_bytes_io = io.BytesIO(); after_image.save(after_bytes_io, format="PNG")

            return ActionVerification(
                success=success,
                # before_state=before_bytes_io.getvalue(), # Omit for now to reduce size
                # after_state=after_bytes_io.getvalue(),
                changes_detected=[element_bounds] if element_bounds else [],
                confidence=float(confidence),
            )
        except Exception as e:
            logger.error(f"Error during action verification: {e}", exc_info=True)
            return None

    async def start(
        self, host: str = "127.0.0.1", port: int = 8000
    ):  # Added host parameter
        """Start MCP server"""
        logger.info(f"Starting OmniMCP server on {host}:{port}")
        # Ensure initial state is loaded? Optional.
        # await self._visual_state.update()
        # logger.info("Initial visual state loaded.")
        await self.mcp.serve(host=host, port=port)  # Use host parameter


# Example for running the server directly (if needed)
# async def main():
#     server = OmniMCP()
#     await server.start()

# if __name__ == "__main__":
#     asyncio.run(main())
