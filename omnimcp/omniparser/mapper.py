# omnimcp/omniparser/mapper.py

from typing import Any  # Added Any

from loguru import logger

# Assuming types are imported correctly
from omnimcp.types import Bounds, UIElement  # Assuming Bounds is tuple (x,y,w,h)


def map_omniparser_to_uielements(
    parser_json: dict, img_width: int, img_height: int
) -> list[UIElement]:
    """Converts raw OmniParser JSON output to a list of UIElement objects."""
    elements: list[UIElement] = []
    element_id_counter = 0
    # Adjust key if needed based on actual OmniParser output schema
    raw_elements: list[dict[str, Any]] = parser_json.get("parsed_content_list", [])

    if not isinstance(raw_elements, list):
        logger.error(
            f"Expected 'parsed_content_list' to be a list, got: {type(raw_elements)}"
        )
        return elements  # Return empty list

    logger.info(f"Processing {len(raw_elements)} raw elements from OmniParser.")

    for item in raw_elements:
        try:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item in parsed_content_list: {item}")
                continue

            # 1. Extract and validate bbox
            bbox_rel = item.get("bbox")
            if not isinstance(bbox_rel, list) or len(bbox_rel) != 4:
                logger.debug(
                    f"Skipping element due to invalid/missing bbox: {item.get('content')}"
                )
                continue  # Skip elements without a valid bbox list

            # 2. Convert bbox to normalized (x, y, width, height) format and validate values
            x_min, y_min, x_max, y_max = bbox_rel
            x = float(x_min)
            y = float(y_min)
            w = float(x_max - x_min)
            h = float(y_max - y_min)

            # Check bounds validity (relative coords, positive w/h)
            # Allow zero coordinates but require positive width/height
            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and w > 0.0
                and h > 0.0
                and (x + w) <= 1.001
                and (y + h) <= 1.001
            ):
                # Add a small tolerance (0.001) for potential floating point inaccuracies near edges
                logger.warning(
                    f"Skipping element due to invalid relative bounds values (x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}): {item.get('content')}"
                )
                continue  # Validate bounds

            # Optionally filter tiny elements based on absolute size
            min_pixel_size = 3  # Minimum width or height in pixels
            if (w * img_width < min_pixel_size) or (h * img_height < min_pixel_size):
                logger.debug(
                    f"Skipping potentially tiny element (w={w * img_width:.1f}, h={h * img_height:.1f} px): {item.get('content')}"
                )
                continue

            bounds: Bounds = (x, y, w, h)

            # 3. Extract and normalize type string
            element_type = str(item.get("type", "unknown")).lower().replace(" ", "_")

            # 4. Extract content
            content = str(item.get("content", ""))

            # 5. Create UIElement
            elements.append(
                UIElement(
                    id=element_id_counter,
                    type=element_type,
                    content=content,
                    bounds=bounds,
                    confidence=float(item.get("confidence", 0.0)),
                    attributes=item.get("attributes", {}) or {},  # Ensure it's a dict
                )
            )
            element_id_counter += 1

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Skipping element due to mapping error: {item.get('content')} - Error: {e}"
            )
        except Exception as unexpected_e:
            # Catch any other unexpected errors during item processing
            logger.error(
                f"Unexpected error mapping element: {item.get('content')} - {unexpected_e}",
                exc_info=True,
            )

    logger.info(
        f"Successfully mapped {len(elements)} UIElements from OmniParser response."
    )
    return elements
