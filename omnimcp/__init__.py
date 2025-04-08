# omnimcp/__init__.py
import sys
from loguru import logger

# Import config first as it might be needed by others
from .config import config

# Now import the setup function from its new location
from .utils import setup_run_logging

# Remove default handler added by loguru at import time
logger.remove()

# --- Initial Setup ---
# Configure base logging (stderr + optional default file)
# This ensures logging works even if AgentExecutor isn't run immediately
if not config.DISABLE_DEFAULT_LOGGING:
    setup_run_logging()  # Call without run_dir to set up defaults
else:
    # If default is disabled, still add stderr at least
    logger.add(
        sys.stderr, level=config.LOG_LEVEL.upper() if config.LOG_LEVEL else "INFO"
    )
    logger.info("Default file logging disabled via config. Stderr logging enabled.")

logger.info(f"OmniMCP package initialized. Log level: {config.LOG_LEVEL.upper()}")

# Optionally expose key classes/functions at the package level
# from .agent_executor import AgentExecutor
# from .visual_state import VisualState
# etc.
