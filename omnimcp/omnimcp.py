# omnimcp/omnimcp.py

"""
OmniMCP: Model Context Protocol for UI Automation through visual understanding.

This module implements the OmniMCP server which provides MCP tools for UI understanding
and interaction. It allows AI models like Claude to observe and interact with user interfaces
through screenshots, element detection, and input simulation.
"""

import io
import time
from typing import List, Optional, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from loguru import logger

from omnimcp.omniparser.client import OmniParserProvider
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
from omnimcp.input import InputController


class VisualState:
    """Manages the current state of visible UI elements."""

    def __init__(self, parser_provider=None):
        """Initialize the visual state manager.

        Args:
            parser_provider: Optional OmniParserProvider instance
        """
        self.elements = []
        self.timestamp = None
        self.screen_dimensions = None
        self._last_screenshot = None
        self._parser = parser_provider or OmniParserProvider()

    async def update(self):
        """Update visual state from screenshot.

        Critical function that maintains screen state.
        """
        # Capture screenshot
        screenshot = take_screenshot()
        self._last_screenshot = screenshot
        self.screen_dimensions = screenshot.size

        # Process with UI parser
        if not self._parser.is_available():
            self._parser.deploy()

        parser_result = self._parser.client.parse_image(screenshot)

        # Update state
        self._update_elements_from_parser(parser_result)
        self.timestamp = time.time()

        return self

    def _update_elements_from_parser(self, parser_result):
        """Process parser results into UIElements."""
        self.elements = []

        if "error" in parser_result:
            logger.error(f"Parser error: {parser_result['error']}")
            return

        for element_data in parser_result.get("parsed_content_list", []):
            ui_element = self._convert_to_ui_element(element_data)
            if ui_element:
                self.elements.append(ui_element)

    def _convert_to_ui_element(self, element_data):
        """Convert parser element to UIElement with normalized coordinates."""
        try:
            # Extract and normalize bounds
            bounds = self._normalize_bounds(element_data.get("bounds", {}))

            # Create UIElement
            return UIElement(
                type=element_data.get("type", "unknown"),
                content=element_data.get("content", ""),
                bounds=bounds,
                confidence=element_data.get("confidence", 0.0),
                attributes=element_data.get("attributes", {}),
            )
        except Exception as e:
            logger.error(f"Error converting element: {e}")
            return None

    def _normalize_bounds(self, bounds_data):
        """Normalize element bounds to 0-1 range."""
        if not bounds_data or not self.screen_dimensions:
            return Bounds(0, 0, 0, 0)

        width, height = self.screen_dimensions

        return Bounds(
            x=bounds_data.get("x", 0) / width,
            y=bounds_data.get("y", 0) / height,
            width=bounds_data.get("width", 0) / width,
            height=bounds_data.get("height", 0) / height,
        )

    def find_element(self, description):
        """Find UI element matching description using semantic matching.

        Critical for action reliability.
        """
        if not self.elements:
            return None

        # Convert current screenshot and elements to a prompt for Claude
        element_descriptions = []
        for i, element in enumerate(self.elements):
            element_descriptions.append(
                f"Element {i}: {element.type} with content '{element.content}' at position {element.bounds}"
            )

        # Create prompt with element descriptions and screenshot
        # elements_str = "\n".join(element_descriptions)
        # prompt = f"""
        # Find the UI element that best matches this description: "{description}"
        #
        # Available elements:
        # {elements_str}
        #
        # Return ONLY the index number of the best matching element. If no good match exists, return -1.
        # """

        # TODO: Implement Claude API call
        # For now, simulate a response by finding the first partial match
        for i, element in enumerate(self.elements):
            if any(
                word in element.content.lower() for word in description.lower().split()
            ):
                return element

        return None


class OmniMCP:
    """Model Context Protocol server for UI understanding."""

    def __init__(self, parser_url: Optional[str] = None, debug: bool = False):
        """Initialize the OmniMCP server.

        Args:
            parser_url: Optional URL for the OmniParser service
            debug: Whether to enable debug mode
        """
        self.input = InputController()
        self.mcp = FastMCP("omnimcp")
        self._visual_state = VisualState(parser_provider=OmniParserProvider(parser_url))
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._debug = debug
        self._debug_context = None
        self._setup_tools()

    def _setup_tools(self):
        """Register MCP tools"""

        @self.mcp.tool()
        async def get_screen_state() -> ScreenState:
            """Get current state of visible UI elements"""
            # Update visual state
            await self._visual_state.update()

            # Return screen state
            return ScreenState(
                elements=self._visual_state.elements,
                dimensions=self._visual_state.screen_dimensions,
                timestamp=self._visual_state.timestamp,
            )

        @self.mcp.tool()
        async def describe_element(description: str) -> str:
            """Get rich description of UI element"""
            # Update visual state
            await self._visual_state.update()

            # Find element
            element = self._visual_state.find_element(description)
            if not element:
                return f"No element found matching: {description}"

            # Generate basic description for now
            # TODO: Enhance with Claude's description
            return (
                f"Found {element.type} with content '{element.content}' "
                f"at position {element.bounds}"
            )

        @self.mcp.tool()
        async def find_elements(query: str, max_results: int = 5) -> List[UIElement]:
            """Find elements matching natural query"""
            # Update visual state
            await self._visual_state.update()

            # For now, use simple matching
            # TODO: Enhance with semantic search
            matching_elements = []
            for element in self._visual_state.elements:
                if any(
                    word in element.content.lower() for word in query.lower().split()
                ):
                    matching_elements.append(element)
                    if len(matching_elements) >= max_results:
                        break

            return matching_elements

        @self.mcp.tool()
        async def click_element(
            description: str,
            click_type: Literal["single", "double", "right"] = "single",
        ) -> InteractionResult:
            """Click UI element matching description"""
            # Update visual state
            await self._visual_state.update()

            # Find element
            element = self._visual_state.find_element(description)
            if not element:
                return InteractionResult(
                    success=False,
                    element=None,
                    error=f"Element not found: {description}",
                )

            # Take before screenshot for verification
            before_screenshot = self._visual_state._last_screenshot

            # Click element using input controller
            success = await self.input.click(element.bounds, click_type)

            # Update visual state after action
            await self._visual_state.update()

            # Verify action
            verification = self._verify_action(
                before_screenshot, self._visual_state._last_screenshot, element.bounds
            )

            return InteractionResult(
                success=success, element=element, verification=verification
            )

        @self.mcp.tool()
        async def scroll_view(
            direction: Literal["up", "down", "left", "right"], amount: float
        ) -> ScrollResult:
            """Scroll in specified direction"""
            # Update visual state
            await self._visual_state.update()

            # Take before screenshot for verification
            before_screenshot = self._visual_state._last_screenshot

            # TODO: Implement scroll using input controller
            # For now, just log
            logger.info(f"Scroll {direction} by {amount}")

            # Update visual state after action
            await self._visual_state.update()

            # Verify action
            verification = self._verify_action(
                before_screenshot, self._visual_state._last_screenshot
            )

            return ScrollResult(
                success=True,
                element=None,
                scroll_amount=amount,
                verification=verification,
            )

        @self.mcp.tool()
        async def type_text(text: str, target: Optional[str] = None) -> TypeResult:
            """Type text, optionally targeting element"""
            # Update visual state
            await self._visual_state.update()

            # If target is provided, click it first
            element = None
            if target:
                click_result = await click_element(target)
                if not click_result.success:
                    return TypeResult(
                        success=False,
                        element=None,
                        error=f"Failed to click target: {target}",
                        text_entered="",
                    )
                element = click_result.element

            # Take before screenshot for verification
            before_screenshot = self._visual_state._last_screenshot

            # Type text using input controller
            success = await self.input.type_text(text)

            # Update visual state after action
            await self._visual_state.update()

            # Verify action
            verification = self._verify_action(
                before_screenshot, self._visual_state._last_screenshot
            )

            return TypeResult(
                success=success,
                element=element,
                text_entered=text,
                verification=verification,
            )

        @self.mcp.tool()
        async def press_key(key: str, modifiers: List[str] = None) -> InteractionResult:
            """Press keyboard key with optional modifiers"""
            # Update visual state
            await self._visual_state.update()

            # Take before screenshot for verification
            before_screenshot = self._visual_state._last_screenshot

            # Press key using input controller
            success = await self.input.press_key(key, modifiers)

            # Update visual state after action
            await self._visual_state.update()

            # Verify action
            verification = self._verify_action(
                before_screenshot, self._visual_state._last_screenshot
            )

            return InteractionResult(
                success=success,
                element=None,
                context={"key": key, "modifiers": modifiers or []},
                verification=verification,
            )

    async def _verify_action(
        self, before_image, after_image, element_bounds=None, action_description=None
    ):
        """Verify action success by comparing before/after screenshots using Claude.

        Args:
            before_image: Screenshot before action
            after_image: Screenshot after action
            element_bounds: Optional bounds to focus verification on
            action_description: Description of the action performed

        Returns:
            ActionVerification object with results
        """
        if not before_image or not after_image:
            return ActionVerification(
                success=False,
                before_state=None,
                after_state=None,
                changes_detected=[],
                confidence=0.0,
            )

        # Convert to bytes for storage
        before_bytes = io.BytesIO()
        after_bytes = io.BytesIO()
        before_image.save(before_bytes, format="PNG")
        after_image.save(after_bytes, format="PNG")

        # Generate diff image
        diff_image = compute_diff(before_image, after_image)

        # Extract region of interest if element_bounds provided
        changes_detected = []

        if element_bounds:
            # Convert normalized bounds to absolute coordinates
            int(element_bounds.x * before_image.width)
            int(element_bounds.y * before_image.height)
            int(element_bounds.width * before_image.width)
            int(element_bounds.height * before_image.height)

            changes_detected.append(element_bounds)

        # TODO: Use Claude Vision API to verify action success
        # Implementation steps:
        # 1. Prepare a prompt that describes the action performed (click, type, etc.)
        # 2. Send the before image, after image, and optionally the diff image to Claude
        # 3. Ask Claude to analyze whether the action was successful by examining UI changes
        # 4. Parse Claude's response to determine success/failure and confidence level
        # 5. Extract any additional context about the changes from Claude's response
        # Example prompt: "I performed [action_description]. Analyze the before and after
        # screenshots and tell me if the action was successful."

        # Placeholder for Claude vision API
        # For now, implement a simple success detection based on pixel changes
        diff_array = np.array(diff_image)
        changes = np.sum(diff_array > 30)  # Threshold for pixel change detection

        # Very basic logic for now
        success = changes > 100  # At least 100 pixels changed
        confidence = min(1.0, changes / (diff_array.size * 0.01)) if success else 0.0

        return ActionVerification(
            success=success,
            before_state=before_bytes.getvalue(),
            after_state=after_bytes.getvalue(),
            changes_detected=changes_detected,
            confidence=float(confidence),
        )

    async def start(self, port: int = 8000):
        """Start MCP server"""
        logger.info(f"Starting OmniMCP server on port {port}")
        await self.mcp.serve(port=port)
