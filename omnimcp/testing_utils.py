# omnimcp/testing_utils.py

"""
Utilities for generating synthetic UI images and test data for OmniMCP tests.
"""

import os
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Assuming types are implicitly available via callers or add specific imports if needed
# from .types import Bounds # Assuming Bounds = Tuple[float, float, float, float]

# Use default font if specific fonts aren't guaranteed in test environment
try:
    # Adjust path if needed, but rely on default if not found
    FONT = ImageFont.truetype("arial.ttf", 15)
except OSError:
    # logger.warning("Arial font not found. Using default PIL font.") # logger might not be configured here
    print("Warning: Arial font not found. Using default PIL font.")
    FONT = ImageFont.load_default()


def generate_test_ui(
    save_path: str | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """
    Generate synthetic UI image with known elements.

    Returns:
        Tuple containing:
            - PIL Image of synthetic UI
            - List of element metadata dictionaries mimicking OmniParser output structure.
    """
    img_width, img_height = 800, 600
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)
    elements = []  # This will be list of DICTS mimicking OmniParser output structure

    # Button
    x1, y1, x2, y2 = 100, 100, 200, 150
    draw.rectangle([(x1, y1), (x2, y2)], fill="blue", outline="black")
    draw.text((110, 115), "Submit", fill="white", font=FONT)
    elements.append(
        {
            "type": "button",
            "content": "Submit",
            "bbox": [
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height,
            ],  # List format [x_min, y_min, x_max, y_max]
            "confidence": 1.0,
        }
    )

    # Text field
    x1, y1, x2, y2 = 300, 100, 500, 150
    draw.rectangle([(x1, y1), (x2, y2)], fill="white", outline="black")
    draw.text((310, 115), "Username", fill="gray", font=FONT)  # Placeholder text
    elements.append(
        {
            "type": "text_field",
            "content": "",  # Actual content usually empty initially
            "bbox": [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height],
            "confidence": 1.0,
            "attributes": {"placeholder": "Username"},
        }
    )

    # Checkbox (unchecked)
    x1, y1, x2, y2 = 100, 200, 120, 220
    draw.rectangle([(x1, y1), (x2, y2)], fill="white", outline="black")
    draw.text((130, 205), "Remember me", fill="black", font=FONT)
    elements.append(
        {
            "type": "checkbox",
            "content": "Remember me",  # Label often associated
            "bbox": [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height],
            "confidence": 1.0,
            "attributes": {"checked": False},
        }
    )

    # Link
    x1_text, y1_text = 400, 200
    link_text = "Forgot password?"
    # Use textbbox to estimate bounds for links/text elements
    try:
        text_bbox = draw.textbbox((x1_text, y1_text), link_text, font=FONT)
        x1, y1, x2, y2 = text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]
    except AttributeError:  # Fallback for older PIL/Pillow without textbbox
        est_w, est_h = 120, 20
        x1, y1 = x1_text, y1_text
        x2, y2 = x1 + est_w, y1 + est_h

    draw.text((x1_text, y1_text), link_text, fill="blue", font=FONT)
    elements.append(
        {
            "type": "link",
            "content": link_text,
            "bbox": [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height],
            "confidence": 1.0,
        }
    )

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)
        print(
            f"Saved synthetic UI image to: {save_path}"
        )  # Use print if logger not setup

    # Returns image and LIST OF DICTS (like OmniParser)
    return img, elements


def generate_action_test_pair(
    action_type: str = "click", target: str = "button", save_dir: str | None = None
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]]]:
    """Generate before/after UI image pair for a specific action."""
    temp_save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        temp_save_path = os.path.join(save_dir, f"before_{action_type}_{target}.png")

    # Uses the generate_test_ui function above
    before_img, elements = generate_test_ui(save_path=temp_save_path)
    after_img = before_img.copy()
    after_draw = ImageDraw.Draw(after_img)

    if action_type == "click" and target == "button":
        after_draw.rectangle([(100, 100), (200, 150)], fill="darkblue", outline="black")
        after_draw.text((110, 115), "Submit", fill="white", font=FONT)
        after_draw.text((100, 170), "Form submitted!", fill="green", font=FONT)
    elif action_type == "type" and target == "text_field":
        after_draw.rectangle([(300, 100), (500, 150)], fill="white", outline="black")
        after_draw.text((310, 115), "testuser", fill="black", font=FONT)
    elif action_type == "check" and target == "checkbox":
        after_draw.rectangle([(100, 200), (120, 220)], fill="white", outline="black")
        after_draw.line([(102, 210), (110, 218)], fill="black", width=2)
        after_draw.line([(110, 218), (118, 202)], fill="black", width=2)
        after_draw.text((130, 205), "Remember me", fill="black", font=FONT)

    if save_dir:
        after_path = os.path.join(save_dir, f"after_{action_type}_{target}.png")
        after_img.save(after_path)
    return before_img, after_img, elements


# Add other necessary helper functions here if they were moved from test files
