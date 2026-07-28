# make_gif.py
import glob
import os
import sys

import fire
from loguru import logger  # Use logger for consistency
from PIL import Image


def create_gif(
    input_dir: str,
    output_name: str = "omnimcp_demo.gif",
    duration_ms: int = 670,  # Default matches -delay 67 (670ms)
    loop: int = 0,  # 0 = loop forever
    optimize: bool = True,  # Try to optimize GIF size
):
    """
    Creates an animated GIF from PNG images in a specified directory,
    ordered by file modification time.

    Args:
        input_dir: Path to the directory containing PNG images.
        output_name: Filename for the output GIF (saved in the current directory).
        duration_ms: Duration (in milliseconds) for each frame.
        loop: Number of loops (0 for infinite).
        optimize: Whether to optimize the GIF palettes and layers.
    """
    logger.info(f"Searching for PNG images in: {input_dir}")

    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find all PNG files
    search_pattern = os.path.join(input_dir, "*.png")
    png_files = glob.glob(search_pattern)

    if not png_files:
        logger.error(f"No PNG files found in directory: {input_dir}")
        sys.exit(1)

    # Sort files by modification time (oldest first)
    try:
        png_files.sort(key=os.path.getmtime)
        logger.info(f"Found {len(png_files)} PNG files, sorted by modification time.")
        # Log first and last few files for verification
        files_to_log = png_files[:3] + (png_files[-3:] if len(png_files) > 3 else [])
        logger.debug(
            f"File order (first/last 3): {[os.path.basename(f) for f in files_to_log]}"
        )
    except Exception as e:
        logger.error(f"Error sorting files by modification time: {e}")
        sys.exit(1)

    # Create list of image objects
    frames: list[Image.Image] = []
    try:
        logger.info("Opening image files...")
        for filename in png_files:
            try:
                img = Image.open(filename)
                # Ensure image is in RGBA or RGB mode for consistency if needed
                # img = img.convert("RGBA") # Uncomment if needed, adds alpha channel
                frames.append(img)
            except Exception as e:
                logger.warning(
                    f"Skipping file {os.path.basename(filename)} due to error: {e}"
                )
                continue  # Skip problematic files

        if not frames:
            logger.error("No valid image frames could be opened.")
            sys.exit(1)

        logger.info(f"Creating GIF '{output_name}' with {len(frames)} frames...")

        # Save as animated GIF
        frames[0].save(
            output_name,
            save_all=True,
            append_images=frames[1:],  # Append remaining frames
            duration=duration_ms,
            loop=loop,
            optimize=optimize,
        )
        logger.success(f"Successfully generated GIF: {output_name}")

    except Exception as e:
        logger.error(f"Failed to create GIF: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Configure logger basic setup if running directly
    # logger.add(sys.stderr, level="INFO") # Example basic config
    fire.Fire(create_gif)
