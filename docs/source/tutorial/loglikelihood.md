# Loglikelihood

## What does Loglikelihood for protein mean?

The protein loglikelihood is a metric which estimates the joint probability of
observing a given sequence of amino-acids. The idea behind such an estimator is to approximate the
probability that a mutated protein will be “natural”, and can effectively be produced by a cell.

These metrics rely on transformers language models .
These models are trained to predict a “masked” amino-acid in a sequence.
As a consequence, they can provide us an estimate of the probability of observing
an amino-acid given the “context” (the surrounding amino-acids).
By multiplying individual probabilities computed for a given amino-acid given its context,
we obtain a pseudo-likelihood, which can be a candidate estimator to approximate a sequence stability.

```python
from biotransformers import BioTransformers

bio_trans = BioTransformers(backend="protbert",multi_gpu=True)

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]


embeddings = bio_trans.compute_loglikelihood(sequences)
```

### Different pass mode

For each provided methods, you can do the compute in a forward mode or in a masked mode. The last one is
longer as we have to mask and compute the probabilities for each masked amino acid.

```python
embeddings = bio_trans.compute_loglikelihood(sequences, pass_mode="masked")
```

### Tokens list

The method give the ability to compute the loglikelihood for only a provided list of amino acids, which will be considered.

```python
embeddings = bio_trans.compute_loglikelihood(sequences, tokens_list=["L","E","R","S","K"])
```
