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
