# omnimcp/visual_state.py

"""
Manages the perceived state of the UI using screenshots, OmniParser,
and element tracking across updates.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from loguru import logger

# Required imports
from omnimcp.config import config
from omnimcp.omniparser.client import OmniParserClient

# Import necessary types and the tracker
from omnimcp.types import Bounds, UIElement, ElementTrack
from omnimcp.tracking import SimpleElementTracker
from omnimcp.utils import take_screenshot, downsample_image


class VisualState:
    """
    Manages the perceived state of the UI using screenshots, OmniParser,
    and element tracking across updates.
    """

    def __init__(self, parser_client: OmniParserClient):
        """Initialize the visual state manager."""
        self._parser_client = parser_client
        if not self._parser_client:
            logger.critical("VisualState initialized without a valid parser_client!")
            raise ValueError("VisualState requires a valid OmniParserClient instance.")

        # State attributes
        self.elements: List[UIElement] = []  # Raw elements from current frame's parse
        self.tracked_elements_view: List[
            ElementTrack
        ] = []  # Tracker's view of elements
        self.timestamp: Optional[float] = (
            None  # Timestamp of the last successful update
        )
        self.screen_dimensions: Optional[Tuple[int, int]] = (
            None  # Original screen dimensions
        )
        self._last_screenshot: Optional[Image.Image] = None  # Original screenshot

        # Internal components
        self.element_tracker = SimpleElementTracker()  # Instantiate the tracker
        self.frame_counter: int = 0  # Track update calls for tracker state

        logger.info("VisualState initialized with SimpleElementTracker.")

    def update(self) -> None:
        """
        Update visual state: capture screen, parse elements via OmniParser,
        and update element tracks. Populates self.elements, self.timestamp,
        self.screen_dimensions, self._last_screenshot, and self.tracked_elements_view.
        """
        self.frame_counter += 1
        logger.info(f"VisualState update requested (Frame: {self.frame_counter})...")
        start_time = time.time()
        screenshot: Optional[Image.Image] = None
        parsed_elements: List[
            UIElement
        ] = []  # Store result before assigning to self.elements

        try:
            # 1. Capture screenshot
            logger.debug("Taking screenshot...")
            screenshot = take_screenshot()
            if screenshot is None:
                raise RuntimeError("Failed to take screenshot.")

            # Store original screenshot and dimensions
            self._last_screenshot = screenshot
            original_dimensions = screenshot.size
            self.screen_dimensions = original_dimensions
            logger.debug(f"Screenshot taken: original dimensions={original_dimensions}")

            # 2. Optionally Downsample before sending to parser
            image_to_parse = screenshot
            scale_factor = config.OMNIPARSER_DOWNSAMPLE_FACTOR
            # Validate factor
            if not (0.0 < scale_factor <= 1.0):
                logger.warning(
                    f"Invalid OMNIPARSER_DOWNSAMPLE_FACTOR: {scale_factor}. Using 1.0."
                )
                scale_factor = 1.0

            if scale_factor < 1.0:
                image_to_parse = downsample_image(screenshot, scale_factor)

            # 3. Process with UI parser client
            if not self._parser_client.server_url:
                logger.error(
                    "OmniParser client server URL not available. Cannot parse."
                )
                self.elements = []
                self.tracked_elements_view = []  # Clear tracks too
                self.timestamp = time.time()
                return  # Exit update early

            logger.debug(
                f"Parsing image (size: {image_to_parse.size}) via {self._parser_client.server_url}..."
            )
            parser_start_time = time.time()
            parser_result = self._parser_client.parse_image(image_to_parse)
            parser_duration = time.time() - parser_start_time
            logger.debug(f"Parsing completed in {parser_duration:.2f}s.")

            # 4. Map parser results to UIElement objects
            logger.debug("Mapping parser results to UIElements...")
            mapping_start_time = time.time()
            # Use helper method to get the list of UIElements for this frame
            parsed_elements = self._parse_and_map_elements(parser_result)
            mapping_duration = time.time() - mapping_start_time
            logger.debug(
                f"Mapped {len(parsed_elements)} elements in {mapping_duration:.2f}s."
            )

            # Assign mapped elements to state for this frame
            self.elements = parsed_elements

            # 5. Update Element Tracker
            logger.debug("Updating element tracker...")
            tracking_start_time = time.time()
            # Pass the newly parsed elements and current frame number
            self.tracked_elements_view = self.element_tracker.update(
                self.elements, self.frame_counter
            )
            tracking_duration = time.time() - tracking_start_time
            logger.info(
                f"Tracker updated in {tracking_duration:.2f}s. Active tracks: {len(self.tracked_elements_view)}"
            )

            # Update timestamp only on full success
            self.timestamp = time.time()
            total_duration = time.time() - start_time
            logger.success(
                f"VisualState update complete for Frame {self.frame_counter}. "
                f"Found {len(self.elements)} raw elements. Active tracks: {len(self.tracked_elements_view)}. "
                f"Total time: {total_duration:.2f}s."
            )

        except Exception as e:
            logger.error(
                f"Failed to update visual state (Frame {self.frame_counter}): {e}",
                exc_info=True,
            )
            # Reset state on failure
            self.elements = []
            self.tracked_elements_view = []  # Also clear tracker view on error
            self.timestamp = time.time()  # Record time of failure
            # Attempt to keep screen dimensions if screenshot was taken
            if screenshot:
                self.screen_dimensions = screenshot.size
            else:
                self.screen_dimensions = None

    def _parse_and_map_elements(self, parser_json: Dict) -> List[UIElement]:
        """
        Helper method to map raw JSON output from OmniParser to UIElement objects.
        Assigns sequential per-frame IDs.
        """
        new_elements: List[UIElement] = []
        element_id_counter = 0  # Assign sequential IDs per-frame

        if not isinstance(parser_json, dict):
            logger.error(
                f"Parser result is not a dictionary: {type(parser_json)}. Cannot map."
            )
            return new_elements
        if "error" in parser_json:
            logger.error(f"Parser returned an error: {parser_json['error']}")
            return new_elements

        raw_elements: List[Dict[str, Any]] = parser_json.get("parsed_content_list", [])
        if not isinstance(raw_elements, list):
            logger.error(
                f"Expected 'parsed_content_list' to be a list, got: {type(raw_elements)}"
            )
            return new_elements

        logger.debug(f"Mapping {len(raw_elements)} raw items from OmniParser.")
        for item in raw_elements:
            # Pass counter to assign ID
            ui_element = self._convert_to_ui_element(item, element_id_counter)
            if ui_element:
                new_elements.append(ui_element)
                element_id_counter += 1  # Increment only for valid elements

        logger.debug(
            f"Successfully mapped {len(new_elements)} valid UIElements for this frame."
        )
        return new_elements

    def _convert_to_ui_element(
        self, item: Dict[str, Any], element_id: int
    ) -> Optional[UIElement]:
        """
        Converts a single item from OmniParser result to a validated UIElement.
        Returns None if item is invalid.
        """
        try:
            if not isinstance(item, dict):
                # logger.warning(f"Skipping non-dict item in parser result: {item}")
                return None  # Silently skip non-dicts

            bbox_rel = item.get("bbox") or item.get("box")  # Check common keys

            if not isinstance(bbox_rel, list) or len(bbox_rel) != 4:
                # logger.debug(f"Skipping element due to invalid/missing bbox: Content='{item.get('content', '')[:50]}...'")
                return None  # Silently skip items without valid bbox structure

            # Attempt conversion, handle potential non-numeric values
            try:
                x_min, y_min, x_max, y_max = map(float, bbox_rel)
            except (ValueError, TypeError) as map_err:
                logger.warning(
                    f"Could not map bbox values to float: {bbox_rel} - Error: {map_err}"
                )
                return None

            # Calculate x, y, w, h from x_min, y_min, x_max, y_max
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            # Validate bounds (relative 0.0 to 1.0, non-negative w/h)
            tolerance = 0.001  # Allow slight float inaccuracies near boundaries
            if not (
                (-tolerance <= x <= 1.0 + tolerance)
                and (-tolerance <= y <= 1.0 + tolerance)
                and (w >= 0.0)  # Check non-negative first
                and (h >= 0.0)
                and (x + w) <= 1.0 + tolerance
                and (y + h) <= 1.0 + tolerance
            ):
                # logger.warning(f"Skipping element with invalid relative bounds: xywh=({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}) Content='{item.get('content', '')[:50]}...'")
                return None  # Silently skip invalid bounds

            # Clamp coordinates strictly between 0.0 and 1.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0 - x, w))  # Clamp width based on clamped x
            h = max(0.0, min(1.0 - y, h))  # Clamp height based on clamped y

            # Filter elements with effectively zero area after clamping
            min_dim_threshold = 1e-5
            if w < min_dim_threshold or h < min_dim_threshold:
                # logger.debug(f"Skipping element with near-zero dimensions after clamping: w={w:.4f}, h={h:.4f}. Content='{item.get('content', '')[:50]}...'")
                return None  # Silently skip zero-area elements

            bounds: Bounds = (x, y, w, h)

            # Filter tiny elements based on absolute pixel size if dimensions available
            if self.screen_dimensions:
                img_width, img_height = self.screen_dimensions
                min_pixel_size = 3  # Minimum width or height in pixels
                if (w * img_width < min_pixel_size) or (
                    h * img_height < min_pixel_size
                ):
                    # logger.debug(f"Skipping tiny element (pixels < {min_pixel_size}): w={w*img_width:.1f}, h={h*img_height:.1f}. Content='{item.get('content', '')[:50]}...'")
                    return None  # Silently skip tiny elements

            # Extract other fields safely
            element_type = (
                str(item.get("type", "unknown")).lower().strip().replace(" ", "_")
            )
            content = str(item.get("content", "")).strip()
            # Ensure confidence is float, default to 0.0 if invalid
            try:
                confidence = float(item.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
            attributes = item.get("attributes", {})
            if not isinstance(attributes, dict):  # Ensure attributes is a dict
                attributes = {}

            # Use the passed-in sequential per-frame ID
            return UIElement(
                id=element_id,
                type=element_type,
                content=content,
                bounds=bounds,
                confidence=confidence,
                attributes=attributes,
            )
        except Exception as unexpected_e:
            # Catch any other unexpected errors during conversion
            logger.error(
                f"Unexpected error mapping element: {item.get('content', '')[:50]}... - {unexpected_e}",
                exc_info=True,
            )
            return None

    def find_element(self, description: str) -> Optional[UIElement]:
        """
        Finds the best matching element using keyword matching with improved scoring
        on the current frame's elements.
        """
        logger.debug(
            f"Finding element by description: '{description}' (using current frame elements: {len(self.elements)})"
        )
        if not self.elements:
            return None

        # Prepare search terms from the description
        search_terms = [term for term in description.lower().split() if term]
        if not search_terms:
            logger.warning("Empty search terms provided to find_element.")
            return None

        best_match: Optional[UIElement] = None
        highest_score: float = (
            -1.0
        )  # Use float, start below any potential positive score

        for element in self.elements:
            content_lower = element.content.lower() if element.content else ""
            type_lower = element.type.lower() if element.type else ""
            current_score: float = 0.0

            # --- Scoring Logic ---
            # 1. Exact Content Match (High Score)
            # Check if the *entire* description matches the element content exactly
            if description.lower() == content_lower:
                current_score += 10.0

            # 2. Term-based scoring
            for term in search_terms:
                # Type Match Bonus (Medium-High Score)
                # Check if term exactly matches the normalized type
                if term == type_lower:
                    current_score += 5.0
                # Check if term matches part of a multi-word type (e.g., "field" in "text_field")
                elif "_" in type_lower and term in type_lower.split("_"):
                    current_score += 2.0  # Lower score for partial type match

                # Content Match Bonuses
                # Check for whole word match in content (Medium Score)
                # Simple split, might need more robust tokenization later
                content_words = content_lower.split()
                if term in content_words:
                    current_score += 3.0
                # Check for substring match in content (Low Score)
                # Award only if whole word didn't match, to avoid double counting
                elif term in content_lower:
                    current_score += 1.0

            # --- Update Best Match ---
            if current_score > highest_score:
                highest_score = current_score
                best_match = element
            # --- Tie-breaking (Optional but can help) ---
            # If scores are equal, maybe prefer smaller elements or specific types?
            # Keep it simple for now: first element with highest score wins.

        # --- Return Result ---
        # Define a minimum score threshold to avoid weak matches
        # Tunable parameter - start with >= 2.0?
        min_score_threshold = 2.0
        if best_match and highest_score >= min_score_threshold:
            logger.info(
                f"Found best match (score={highest_score:.1f}) for '{description}': "
                f"ID={best_match.id}, Type={best_match.type}, Content='{best_match.content[:30]}...'"
            )
            return best_match
        else:
            logger.warning(
                f"No suitable element found with score >= {min_score_threshold} for: '{description}' (Highest score: {highest_score:.1f})"
            )
            return None  # Explicitly return None if no good match found
