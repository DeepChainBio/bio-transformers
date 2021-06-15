# Quick Start

## Display available backend

```python
from biotransformers import BioTransformers
BioTransformers.list_backend()

    *   esm1_t34_670M_UR100
    *   esm1_t6_43M_UR50S
    *   esm1b_t33_650M_UR50S
    *   esm_msa1_t12_100M_UR50S
    *   protbert
    *   protbert_bfd
```

## Compute embeddings on gpu
The multi-gpus option will use pytorch nn.DataParallel module to use multiple embeddings for the inference.
All the GPUs available are used.

```python
sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "RSKEPVSGFDLIRDHISQTGMPPTRAEIARSKEPVSGRKGVIEIVSGASRGIRLLQEE",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ]

bio_trans = BioTransformers(backend="protbert",multi_gpu=True,batch_size=2)
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'))

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

where:

 - pooling_list: kind of aggregation functions to be used. 'cls' return the `<CLS>` token embedding used for classification. 'mean' will make the mean of all the token a sequence.
