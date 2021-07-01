# Loglikelihood

The protein loglikelihood is a metric which estimates the joint probability of
observing a given sequence of amino-acids. The idea behind such an estimator is to approximate the
probability that a mutated protein will be “natural”, and can effectively be produced by a cell.

These metrics rely on transformers language model. These models are trained to predict a “masked” amino-acid in a sequence.
As a consequence, they can provide us an estimate of the probability of observing an amino-acid given the “context” (the surrounding amino-acids).
By multiplying individual probabilities computed for a given amino-acid given its context, we obtain a pseudo-likelihood, which can be a candidate estimator to approximate a sequence stability.

```python
from biotransformers import BioTransformers
import ray

ray.init()
bio_trans = BioTransformers(backend="protbert",num_gpus=2)

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "RQQEVFDLIQQEVFDLIQQEVFIRDAQRLGFRQQEVFDLIRDHISQTGMPPTRAALARKGVIEIVSGASRGIRLLQEE",
        "QEEVFDLIQQEVFDLIRDHISQTGMPPTRAMPPTRAEIAQQARKGVIEIVSGASRGIRLLQEE"
    ]

loglikelihood = bio_trans.compute_loglikelihood(sequences, batch_size=2)
```

## Different pass mode

For each provided methods, you can do the compute in a ``forward`` mode or in a ``masked`` mode. The last one is
longer as we have to mask and compute the probabilities for each masked amino acid.

```python
embeddings = bio_trans.compute_loglikelihood(sequences, pass_mode="masked", batch_size=2)
```

## Tokens list

The method give the ability to compute the loglikelihood for only a provided list of amino acids, which will be considered.

```python
UNNATURAL = list("ACDEFGHIKLMNPQRSTVWY") + ["-"]
loglikelihood = bio_trans.compute_loglikelihood(sequences, tokens_list=UNNATURAL)
```

## Probabilities

The ``compute_loglikelihoods`` relies on the ``compute_probabilities`` function.

This last function will compute for each amino acids position in the sequence the a dictionnary where keys represent the natural amino acids, and values the probabilities to be at the position.

For example:

```python
from biotransformers import BioTransformers

bio_trans = BioTransformers(backend="protbert",num_gpus=1)

sequence = ["MKT"]
probabilities = bio_trans.compute_(sequence, batch_size=1)

print(probabilities)
```

```python
>>
[{0: {'L': 0.06550145316598321, 'A': 0.021559458419220974, 'G': 0.029741129950678777, 'V': 0.0329506745800003, 'E': 0.03389950500319548, 'S': 0.10401323529266542, 'I': 0.04399518228657259, 'K': 0.1534323153578508, 'R': 0.08616676439914424, 'D': 0.010983572050921635, 'T': 0.04474224433539647, 'P': 0.01569993609938641, 'N': 0.027836286891774507, 'Q': 0.037557728840479546, 'F': 0.020606235301203788, 'Y': 0.01243454224917041, 'M': 0.21207524064947852, 'H': 0.015025274369047291, 'C': 0.013031914446968728, 'W': 0.018747306310860856},

 1: {'L': 0.03176897920072879, 'A': 0.013685848027567242, 'G': 0.01709074216275199, 'V': 0.018786360542915624, 'E': 0.016411511761942357, 'S': 0.02157161007259761, 'I': 0.019570515195473124, 'K': 0.026416232407458887, 'R': 0.021930249525274396, 'D': 0.008674132240173953, 'T': 0.018818536773492975, 'P': 0.010970933229272459, 'N': 0.01349720693939123, 'Q': 0.014703372924399499, 'F': 0.010715260172378251, 'Y': 0.00931640096204737, 'M': 0.7010288899792522, 'H': 0.009361870192728095, 'C': 0.007965577806480653, 'W': 0.007715769883673336},

  2: {'L': 0.07383247230045219, 'A': 0.03555995965068629, 'G': 0.03454727111803637, 'V': 0.043748770514437235, 'E': 0.04069625263096508, 'S': 0.06924489597284503, 'I': 0.046173613390643166, 'K': 0.2299759248798167, 'R': 0.06749564661032614, 'D': 0.0224069594369746, 'T': 0.03940009938504622, 'P': 0.02301058203142933, 'N': 0.03441775848661052, 'Q': 0.04373499771477881, 'F': 0.028093375324345762, 'Y': 0.02461900744880924, 'M': 0.025029056199102815, 'H': 0.0818692944874724, 'C': 0.016498739542946495, 'W': 0.01964532287427556}}]
```

For each position, we have  0,1,2 which correpond to amino acids M,K,T, we have a dictionnary of probabilities for each natural amino acids.
