ESM_LIST = [
    # "esm1_t34_670M_UR50S",
    # "esm1_t34_670M_UR50D",
    "esm1_t34_670M_UR100",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
    "esm1b_t33_650M_UR50S",
    "esm_msa1_t12_100M_UR50S",
    "esm_msa1b_t12_100M_UR50S",
    "esm1v_t33_650M_UR90S_1",
]

ROSTLAB_LIST = ["Rostlab/prot_bert", "Rostlab/prot_bert_bfd"]

MAPPING_PROTBERT = {
    "protbert": "Rostlab/prot_bert",
    "protbert_bfd": "Rostlab/prot_bert_bfd",
}

DEFAULT_ESM_MODEL = "esm1_t34_670M_UR100"
DEFAULT_ROSTLAB_MODEL = "Rostlab/prot_bert"

BACKEND_LIST = ESM_LIST + list(MAPPING_PROTBERT.keys())

NATURAL_AAS_LIST = list("ACDEFGHIKLMNPQRSTVWY")
