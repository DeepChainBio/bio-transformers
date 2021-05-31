"""Main module to build either ESM or protbert model"""

from biotransformers.utils.constant import BACKEND_LIST, MAPPING_PROTBERT
from biotransformers.utils.utils import format_backend
from biotransformers.wrappers.esm_wrappers import ESMWrapper
from biotransformers.wrappers.rostlab_wrapper import RostlabWrapper


class BioTransformers:
    """
    General class to choose an ESM or ProtBert backend
    Abstract method are implemented in transformers
    """

    def __init__(
        self,
        backend: str = "esm1_t6_43M_UR50S",
        device: str = None,
        multi_gpu: bool = False,
    ):
        pass

    def __new__(
        cls,
        backend: str = "esm1_t6_43M_UR50S",
        device: str = None,
        multi_gpu: bool = False,
    ):
        format_list = "\n".join(format_backend(BACKEND_LIST))
        assert backend in BACKEND_LIST, f"Choose backend in \n\n{format_list}"

        if backend.__contains__("esm"):
            return ESMWrapper(backend, device=device, multi_gpu=multi_gpu)
        else:
            return RostlabWrapper(
                MAPPING_PROTBERT[backend], device=device, multi_gpu=multi_gpu
            )

    @staticmethod
    def list_backend() -> None:
        """Get all possible backend for the model"""
        print(
            "Use backend in this list :\n\n",
            "\n".join(format_backend(BACKEND_LIST)),
            sep="",
        )
