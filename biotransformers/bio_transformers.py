"""Main module to build either ESM or protbert model"""

from typing import List

from .esm_wrappers import ESMWrapper, esm_list
from .rostlab_wrapper import RostlabWrapper, rostlab_list

MODEL_LIST = esm_list + rostlab_list


class BioTransformers:
    """
    General class to choose an ESM or ProtBert backend
    Abstract method are implemented in transformers
    """

    def __init__(self, device: str, model_dir: str = "esm1_t6_43M_UR50S"):
        pass

    def __new__(cls, device: str = None, model_dir: str = "esm1_t6_43M_UR50S"):
        if model_dir.__contains__("esm"):
            instance = ESMWrapper(model_dir, device=device)
        else:
            instance = RostlabWrapper(model_dir, device=device)
        return instance

    @staticmethod
    def list_backend() -> None:
        """Get all possible backend for the model"""
        print(
            "Use backend in this list :\n\n",
            "\n".join(format_backend(MODEL_LIST)),
            sep="",
        )


def format_backend(backend_list: List[str]) -> List[str]:
    """format of list to display"""
    return ["  *" + " " * 3 + model for model in backend_list]
