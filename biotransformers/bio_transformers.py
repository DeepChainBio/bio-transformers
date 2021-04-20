"""Main module to build either ESM or protbert model"""

from typing import List

from .esm_wrappers import ESMWrapper, esm_list
from .rostlab_wrapper import RostlabWrapper

MAPPING_PROTBERT = {
    "protbert": "Rostlab/prot_bert",
    "protbert_bfd": "Rostlab/prot_bert_bfd",
}
BACKEND_LIST = esm_list + list(MAPPING_PROTBERT.keys())


class BioTransformers:
    """
    General class to choose an ESM or ProtBert backend
    Abstract method are implemented in transformers
    """

    def __init__(
        self,
        backend: str = "esm1_t6_43M_UR50S",
        device: str = None,
    ):
        pass

    def __new__(
        cls,
        backend: str = "esm1_t6_43M_UR50S",
        device: str = None,
    ):
        format_list = "\n".join(format_backend(BACKEND_LIST))
        assert backend in BACKEND_LIST, f"Choose backend in \n\n{format_list}"

        if backend.__contains__("esm"):
            instance = ESMWrapper(backend, device=device)
        else:
            instance = RostlabWrapper(MAPPING_PROTBERT[backend], device=device)
        return instance

    @staticmethod
    def list_backend() -> None:
        """Get all possible backend for the model"""
        print(
            "Use backend in this list :\n\n",
            "\n".join(format_backend(BACKEND_LIST)),
            sep="",
        )


def format_backend(backend_list: List[str]) -> List[str]:
    """format of list to display"""
    return ["  *" + " " * 3 + model for model in backend_list]
