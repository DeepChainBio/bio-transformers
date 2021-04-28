import logging
import os


def logger(module_name: str):
    """Basic python logger"""
    if module_name.endswith("py"):
        module_name = os.path.splitext(module_name)[0]

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
