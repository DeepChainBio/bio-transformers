<p align="center">
  <img width="50%" src="./.source/_static/deepchain.png">
</p>


![PyPI](https://img.shields.io/pypi/v/bio-transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![Documentation Status](https://readthedocs.org/projects/bio-transformers/badge/?version=latest)](https://bio-transformers.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/delfosseaurelien/bio-transformers/branch/main/graph/badge.svg?token=URROG4GV2C)](https://codecov.io/gh/delfosseaurelien/bio-transformers)

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
bio-transformers is a python wrapper on top of the **ESM/Protbert** models, which are **Transformers protein language models**, trained on millions of proteins and used to predict embeddings.
This package provides a unified interface to use all these models - which we call `backends`. For instance you'll be able to compute natural amino-acids probabilities or embeddings on multiple-GPUs.

 You can find the original repositories for the models here :
 - [ESM](https://github.com/facebookresearch/esm/)
 - [Protbert](https://github.com/agemagician/ProtTrans)

## Installation
It is recommended to work with conda environments in order to manage the specific dependencies of this package.
```bash
  conda create --name bio-transformers python=3.7 -y
  conda activate bio-transformers
  pip install bio-transformers
```
# Usage

## Quick start
The main class ```BioTranformers``` allows developers to use Protbert and ESM backends

```python
> from biotransformers import BioTransformers
> BioTransformers.list_backend()
```
```
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
By default, the ```compute_embeddings``` function returns the ```<CLS>``` token embeddings.
You can add a ```pool_mode``` in addition, so you can compute the mean of the tokens embeddings.

```python
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(backend="protbert")
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'))

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

### Multi-gpu
If you have access to multiple GPUs, you can activate the ```multi_gpu``` option to speed-up the inference.
This option relies on ```torch.nn.DataParallel```.
```python
bio_trans = BioTransformers(backend="protbert",multi_gpu=True)
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'))
```


## Pseudo-Loglikelihood
The protein loglikelihood is a metric that estimates the joint probability of observing a given sequence of amino acids. The idea behind such an estimator is to approximate the probability that a mutated protein will be “natural”, and can effectively be produced by a cell.

These metrics rely on transformers language models. These models are trained to predict a “masked” amino acid in a sequence. As a consequence, they can provide us with an estimate of the probability of observing an amino acid given the “context” (the surrounding amino acids).  By multiplying individual probabilities computed for a given amino-acid given its context, we obtain a pseudo-likelihood, which can be a candidate estimator to approximate sequence stability.
```python
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(backend="protbert",device="cuda:0")
loglikelihood = bio_trans.compute_loglikelihood(sequences)
```

# Roadmap:
  - support MSA transformers
  - add compute_accuracy functionnality
  - support finetuning of model with multiple-gpus

# Citations
Here some papers on interest on the subject.

The excellent ProtBert work can be found at [(biorxiv preprint)](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3.full.pdf):

```bibtex
@article{protTrans2021,
  author={Ahmed Elnaggar and Michael Heinzinger, Christian Dallago1,Ghalia Rihawi, Yu Wang, Llion Jones, Tom Gibbs, Tamas Feher, Christoph Angerer,Debsindhu Bhowmik and Burkhard Rost},
  title={ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Deep Learning and High Performance Computing},
  year={2019},
  doi={10.1101/2020.07.12.199554},
  url={https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3.full.pdf},
  journal={bioRxiv}
}
```

For the ESM model, see [(biorxiv preprint)](https://www.biorxiv.org/content/10.1101/622803v4):
```bibtex
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={bioRxiv}
}
```

For the self-attention contact prediction, see [the following paper (biorxiv preprint)](https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1):

```bibtex
@article{rao2020transformer,
  author = {Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander},
  title={Transformer protein language models are unsupervised structure learners},
  year={2020},
  doi={10.1101/2020.12.15.422761},
  url={https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1},
  journal={bioRxiv}
}
```

For the MSA Transformer, see [the following paper (biorxiv preprint)](https://doi.org/10.1101/2021.02.12.430858):

```bibtex
@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1},
  journal={bioRxiv}
}
```



# License

This source code is licensed under the **Apache 2** license found in the `LICENSE` file in the root directory.
