"""
Synthetic UI testing for OmniMCP.

This module provides utilities for testing OmniMCP using programmatically
generated UI images instead of relying on real displays.
"""

import os
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Any, Optional


def generate_test_ui(
    save_path: Optional[str] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Generate synthetic UI image with known elements.

    Args:
        save_path: Optional path to save the generated image for review

    Returns:
        Tuple containing:
            - PIL Image of synthetic UI
            - List of element metadata dictionaries
    """
    # Create blank canvas
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Draw UI elements with known positions
    elements = []

    # Button
    draw.rectangle([(100, 100), (200, 150)], fill="blue", outline="black")
    draw.text((110, 115), "Submit", fill="white")
    elements.append(
        {
            "type": "button",
            "content": "Submit",
            "bounds": {
                "x": 100 / 800,
                "y": 100 / 600,
                "width": 100 / 800,
                "height": 50 / 600,
            },
            "confidence": 1.0,
        }
    )

    # Text field
    draw.rectangle([(300, 100), (500, 150)], fill="white", outline="black")
    draw.text((310, 115), "Username", fill="gray")
    elements.append(
        {
            "type": "text_field",
            "content": "Username",
            "bounds": {
                "x": 300 / 800,
                "y": 100 / 600,
                "width": 200 / 800,
                "height": 50 / 600,
            },
            "confidence": 1.0,
        }
    )

    # Checkbox (unchecked)
    draw.rectangle([(100, 200), (120, 220)], fill="white", outline="black")
    draw.text((130, 205), "Remember me", fill="black")
    elements.append(
        {
            "type": "checkbox",
            "content": "Remember me",
            "bounds": {
                "x": 100 / 800,
                "y": 200 / 600,
                "width": 20 / 800,
                "height": 20 / 600,
            },
            "confidence": 1.0,
        }
    )

    # Link
    draw.text((400, 200), "Forgot password?", fill="blue")
    elements.append(
        {
            "type": "link",
            "content": "Forgot password?",
            "bounds": {
                "x": 400 / 800,
                "y": 200 / 600,
                "width": 120 / 800,
                "height": 20 / 600,
            },
            "confidence": 1.0,
        }
    )

    # Save the image if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)

    return img, elements


def generate_action_test_pair(
    action_type: str = "click", target: str = "button", save_dir: Optional[str] = None
) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]]]:
    """Generate before/after UI image pair for a specific action.

    Args:
        action_type: Type of action ("click", "type", "check")
        target: Target element type ("button", "text_field", "checkbox")
        save_dir: Optional directory to save before/after images for review

    Returns:
        Tuple containing:
            - Before image
            - After image showing the effect of the action
            - List of element metadata
    """
    # Use a temporary path if we need to save both images
    temp_save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        temp_save_path = os.path.join(save_dir, f"before_{action_type}_{target}.png")

    before_img, elements = generate_test_ui(save_path=temp_save_path)
    after_img = before_img.copy()
    after_draw = ImageDraw.Draw(after_img)

    if action_type == "click" and target == "button":
        # Show button in pressed state
        after_draw.rectangle([(100, 100), (200, 150)], fill="darkblue", outline="black")
        after_draw.text((110, 115), "Submit", fill="white")
        # Add success message
        after_draw.text((100, 170), "Form submitted!", fill="green")

    elif action_type == "type" and target == "text_field":
        # Show text entered in field
        after_draw.rectangle([(300, 100), (500, 150)], fill="white", outline="black")
        after_draw.text((310, 115), "testuser", fill="black")

    elif action_type == "check" and target == "checkbox":
        # Show checked checkbox
        after_draw.rectangle([(100, 200), (120, 220)], fill="white", outline="black")
        after_draw.line([(102, 210), (110, 218)], fill="black", width=2)
        after_draw.line([(110, 218), (118, 202)], fill="black", width=2)
        after_draw.text((130, 205), "Remember me", fill="black")

    # Save the after image if requested
    if save_dir:
        after_path = os.path.join(save_dir, f"after_{action_type}_{target}.png")
        after_img.save(after_path)

    return before_img, after_img, elements


def save_all_test_images(output_dir: str = "test_images"):
    """Save all test images to disk for manual inspection.

    Args:
        output_dir: Directory to save images to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save basic UI
    ui_img, elements = generate_test_ui(
        save_path=os.path.join(output_dir, "synthetic_ui.png")
    )

    # Define verified working action-target combinations
    verified_working = [
        # These combinations have been verified to produce different before/after images
        ("click", "button"),  # Click submit button shows success message
        ("type", "text_field"),  # Type in username field
        ("check", "checkbox"),  # Check the remember me box
    ]

    # TODO: Fix and test these combinations:
    # ("click", "checkbox"), # Click to check checkbox
    # ("click", "link"),     # Click link to show as visited

    # Save action pairs for working combinations
    for action, target in verified_working:
        try:
            before, after, _ = generate_action_test_pair(action, target)

            # Save before image
            before_path = os.path.join(output_dir, f"before_{action}_{target}.png")
            before.save(before_path)

            # Save after image
            after_path = os.path.join(output_dir, f"after_{action}_{target}.png")
            after.save(after_path)

            print(f"Generated {action} on {target} images")
        except Exception as e:
            print(f"Error generating {action} on {target}: {e}")


def create_element_overlay_image(save_path: Optional[str] = None) -> Image.Image:
    """Create an image with UI elements highlighted and labeled for human review.

    Args:
        save_path: Optional path to save the visualization

    Returns:
        PIL Image with element visualization
    """
    img, elements = generate_test_ui()
    draw = ImageDraw.Draw(img)

    # Draw bounding box and label for each element
    for i, element in enumerate(elements):
        bounds = element["bounds"]

        # Convert normalized bounds to absolute coordinates
        x = int(bounds["x"] * 800)
        y = int(bounds["y"] * 600)
        width = int(bounds["width"] * 800)
        height = int(bounds["height"] * 600)

        # Draw a semi-transparent highlight box
        highlight = Image.new("RGBA", (width, height), (255, 255, 0, 128))
        img.paste(highlight, (x, y), highlight)

        # Draw label
        draw.text(
            (x, y - 15),
            f"{i}: {element['type']} - '{element['content']}'",
            fill="black",
        )

    # Save the image if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)

    return img


if __name__ == "__main__":
    # Generate and save test images when run directly
    save_all_test_images()

    # Create and save element visualization
    create_element_overlay_image(save_path="test_images/elements_overlay.png")

    print("Test images saved to 'test_images/' directory")
