<p align="center">
  <img width="50%" src="./.source/_static/deepchain.png">
</p>


![PyPI](https://img.shields.io/pypi/v/bio-transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)

<details><summary>Table of contents</summary>

- [Description](#bio-transformers)
- [Installation](#Installation)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Compute embeddings](#embeddings)
  - [Pseudo-Loglikelihood](#pseudo-loglikelihood)
- [Roadmap](#roadmap)
- [Citations](#citations)
- [License](#license)
</details>

# Bio-transformers
bio-transformers is a python wrapper on top of the **ESM/Protbert** model, which are **Transformers protein language model**, trained on millions on proteins and used to predict embeddings.
This package provide other functionalities (like compute the loglikelihood of a protein) or compute embeddings on multiple-gpu.

 You can find the original repo here :
 - [ESM](https://github.com/facebookresearch/esm/)
 - [Protbert](https://github.com/agemagician/ProtTrans)

## Installation
It is recommended to work with conda environnements in order to manage the specific dependencies of the package.
```bash
  conda create --name bio-transformers python=3.7 -y
  conda activate bio-transformers
  pip install bio-transformers
```
# Usage

## Quick start
The main class ```BioTranformers``` allow the developper to use Protbert and ESM backend

```
>>from biotransformers import BioTransformers
>>BioTransformers.list_backend()
Use backend in this list :

  *   esm1_t34_670M_UR100
  *   esm1_t6_43M_UR50S
  *   esm1b_t33_650M_UR50S
  *   esm_msa1_t12_100M_UR50S
  *   protbert
  *   protbert_bfd

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

bio_trans = BioTransformers(backend="protbert")
embeddings = bio_trans.compute_embeddings(sequences, pooling_list=['mean'])

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

## Pseudo-Loglikelihood
The protein loglikelihood is a metric which estimates the joint probability of observing a given sequence of amino-acids. The idea behind such an estimator is to approximate the probability that a mutated protein will be “natural”, and can effectively be produced by a cell.

These metrics rely on transformers language models . These models are trained to predict a “masked” amino-acid in a sequence. As a consequence, they can provide us an estimate of the probability of observing an amino-acid given the “context” (the surrounding amino-acids).  By multiplying individual probabilities computed for a given amino-acid given its context, we obtain a pseudo-likelihood, which can be a candidate estimator to approximate a sequence stability.
```
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(backend="protbert",device="cuda:0")
loglikelihood = bio_trans.compute_loglikelihood(sequences)
```

# Roadmap:
  - Support multi-gpu forward
  - support MSA transformers
  - add compute_accuracy functionnality
  - support finetuning of model

# Citations

# License

This source code is licensed under the **Apache 2** license found in the `LICENSE` file in the root directory.
