import logging
import os


# ==========================================================
# CREATE LOG DIRECTORY
# ==========================================================

os.makedirs(
    "logs",
    exist_ok=True
)


# ==========================================================
# LOG FILE
# ==========================================================

LOG_FILE = "logs/pipeline.log"


# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(name)s - "
        "%(message)s"
    ),
    encoding="utf-8"
)


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(
    __name__
)