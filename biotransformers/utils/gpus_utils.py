from typing import Tuple

import torch
from biotransformers.utils.logger import logger

log = logger("gpus_utils")


def set_device(device: str, multi_gpu: bool) -> Tuple[str, bool]:
    """Set the correct device CPU/GPU

    Args:
        device (str) : could be cpu/cuda:0/cuda
        multi_gpu (bool) : use multi_gpu the same Node

    Returns:
        Tuple[str, bool]:
            * device: str
            * multi_gpu: bool
    """
    n_gpus = torch.cuda.device_count()
    if multi_gpu:
        if not torch.cuda.is_available():
            log.warning("No GPU available, use CPU device")
            return "cpu", False

        if not n_gpus > 1:
            log.warning("Trying to use multi-gpu with only one device, use cuda:0")
            return "cuda:0", False
        else:
            log.info("GPU available, use multi-gpu with %s devices" % n_gpus)
            return "cuda", True

    if device is not None:
        if "cuda" in device:
            if not torch.cuda.is_available():
                log.warning("No GPU available, use CPU device.")
                return "cpu", False
            else:
                log.info("GPU available, use device %s." % device)
                return device, False
        else:
            log.info("Use cpu device.")
            return "cpu", False
    else:
        if torch.cuda.is_available():
            log.info("GPU available, use cuda:0 device")
            return "cuda:0", False
        else:
            log.info("Use cpu device.")
            return "cpu", False
