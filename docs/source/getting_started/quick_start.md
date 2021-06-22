# Quick Start

## Display available backend

```python
from biotransformers import BioTransformers
BioTransformers.list_backend()

>>
    *   esm1_t34_670M_UR100
    *   esm1_t6_43M_UR50S
    *   esm1b_t33_650M_UR50S
    *   esm_msa1_t12_100M_UR50S
    *   protbert
    *   protbert_bfd
```

## Compute embeddings on gpu

Please refer to the [multi-gpus section](https://bio-transformers.readthedocs.io/en/develop/documentation/multi_gpus.html) to have a full understanding of the functionnality.

```python
import ray

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "RSKEPVSGFDLIRDHISQTGMPPTRAEIARSKEPVSGRKGVIEIVSGASRGIRLLQEE",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ]

ray.init()
bio_trans = BioTransformers(backend="protbert", num_gpus=4)
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'))

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

where:

- pooling_list: kind of aggregation functions to be used. 'cls' return the `<CLS>` token embedding used for classification. 'mean' will make the mean of all the tokens a sequence.
