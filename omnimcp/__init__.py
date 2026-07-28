import os
import sys

from loguru import logger

from omnimcp.config import config

# Remove default handler
logger.remove()

# Add stderr handler (keep this functionality)
logger.add(sys.stderr, level=config.LOG_LEVEL.upper() if config.LOG_LEVEL else "INFO")


# Define a function to configure run-specific logging
def setup_run_logging(run_dir=None):
    """
    Configure additional logging for a specific run.

    Args:
        run_dir: Directory to store run-specific logs. If None, logs go to default logs directory.

    Returns:
        The log file path
    """
    # Determine log file location
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        log_file_path = os.path.join(run_dir, "run.log")
    else:
        log_dir = config.LOG_DIR or "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "run_{time:YYYY-MM-DD_HH-mm-ss}.log")

    # Add run-specific log handler
    logger.add(
        log_file_path, rotation="50 MB", level="DEBUG", encoding="utf8", enqueue=True
    )

    logger.info(f"Run logging configured. Log path: {log_file_path}")
    return log_file_path


# Set up default logging (for non-run use)
if not config.DISABLE_DEFAULT_LOGGING:
    setup_run_logging()
