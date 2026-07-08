import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured to write structured logs to stdout.
    Render captures stdout/stderr and surfaces them in the Log Stream tab.
    Format: [LEVEL]  YYYY-MM-DD HH:MM:SS  name  message
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated imports
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(levelname)-8s  %(asctime)s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    # Prevent log records from propagating to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger
