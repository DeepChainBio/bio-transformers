"""This module build a general logger module"""
import logging
import os


def logger(module_name: str) -> logging.Logger:
    """Configure the logger with formatter and handlers.

    The log level depends on the environment variable `BIO_LOG_LEVEL`.

    - 0: NOTSET, will be set to DEBUG
    - 1: DEBUG
    - 2: INFO (default)
    - 3: WARNING
    - 4: ERROR
    - 5: CRITICAL
    https://docs.python.org/3/library/logging.html#levels

    Args:
        module_name (str): module name

    Returns:
        [Logger]: instantiate logger object
    """
    if module_name.endswith("py"):
        module_name = os.path.splitext(module_name)[0]

    logger_ = logging.getLogger(module_name)
    logger_.propagate = False
    log_level = os.environ.get("BIO_LOG_LEVEL", "2")
    log_level_int = max(int(log_level) * 10, 10)
    logger_.setLevel(log_level_int)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(log_level_int)
    logger_.addHandler(handler)

    return logger_
