<p align="center">
  <img width="50%" src="./.source/_static/deepchain.png">
</p>


# Description
bio-transformers is a wrapper on top of the ESM/Protbert model, trained on millions on proteins and used to predict embeddings.
This package provide other functionalities (like compute the loglikelihood of a protein) or compute embeddings on multiple-gpu.

## Installation
It is recommended to work with conda environnements in order to manage the specific dependencies of the package.
```bash
  conda create --name bio-transformers python=3.7 -y 
  conda activate bio-transformers
  pip install bio-transformers
```

# How it works
The main class ```BioTranformers``` allow the developper to use Protbert and ESM backend

```
from biotransformers import BioTransformers
BioTransformers.list_backend()
```

## Embeddings
Choose a backend and pass a list of sequences of Amino acids to compute the embeddings.
By default, the ```compute_embeddings``` function return the ```<CLS>``` token embedding.
You can add a ```pooling_list``` in addition , so you can compute the mean of the tokens embeddings.

```
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(model_dir="Rostlab/prot_bert")
embeddings = bio_trans.compute_embeddings(sequences, pooling_list=['mean'])

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

## Loglikelihood
Choose a backend and pass a list of sequences of Amino acids to compute the Loglikelihood.

```
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(model_dir="Rostlab/prot_bert")
loglikelihood = bio_trans.compute_loglikelihood(sequences)
```