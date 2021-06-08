# Finetuning

## How to finetune a model?
`bio-transformers` uses pytorch-lightning to easily load pre-trained model and finetune it on your own datasets. The method `train_masked` automatically scale on your visible GPU to train in parallel thanks to the different accelerator.

It is strongly recommended to use the `DDP` accelerator for training : [ddp](https://pytorch.org/docs/stable/notes/ddp.html). You should know that `DDP` will launch several python instances, as a consequence, a model should be finetuned in a separate script, and not be mixed with inference function like `compute_loglikelihood` or `compute_embeddings` to avoid GPU conflicts.

The model will be finetuned randomly by masking a proportion of amino acid in a sequence it commonly does in most state of the art paper.

## Caution

This method is developed to be runned on GPU, please take care to have the proper CUDA installation.

Do not train model `DDP` **accelerator** in a notebook. Do not mix training and compute inference function like `compute_accuracy` or `compute_loglikelihood`  in the same script except with `DP` acceletator.
 With `DDP`, load the finetune model in a separate script like below.

```python
from biotransformers import BioTransformers

bio_trans = BioTransformers("esm1_t6_43M_UR50S", device="cuda", multi_gpu=True)
bio_trans.load_model("logs/finetune_masked/version_X/esm1_t6_43M_UR50S_finetuned.pt")
acc_after = bio_trans.compute_accuracy(..., batch_size=32)
```

## Parameters
The function can handle a fasta file or a list of sequences directly:

 - **train_sequences**: Could be a list of sequence of a the path of a fasta files with SeqRecords.

Seven arguments are important for the training:
 - **lr**: the default learning rate (keep it low : <5e10-4)
 - **warmup_updates**:  the number of step (not epochs, optimizer step) to do while increasing the leraning rate from a **warmup_init_lr** to **lr**.
- **epochs** :  number of epoch for training. Defaults to 10.
- **batch_size** :  This size is only uses internally to compute the **accumulate_grad_batches** for gradient accumulation (TO BE UPDATED). The **toks_per_batch** will dynamically determine the number of sequences in a batch, in order to avoir GPU saturation.
- **acc_batch_size** : Number of batch to consider befor computing gradient.

Three arguments allow to custom the masking function used for building the training dataset:

- **masking_ratio** : ratio of tokens to be masked. Defaults to 0.025.
- **random_token_prob** : the probability that the chose token is replaced with a random token.
- **masking_prob**: the probability that the chose token is replaced with a mask token.

All the results will be saved in logs directory:

- **logs_save_dir**: Defaults directory to logs.
- **logs_name_exp**: Name of the experience in the logs.
- **checkpoint**: Path to a checkpoint file to restore training session.
- **save_last_checkpoint**: Save last checkpoint and 2 best trainings models
to restore the training session. Take a large amount of time and memory.

## Example : training script

Training on some swissprot sequences. Training only works on GPU.

```python
import biodatasets
import numpy as np
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train on small sequence
length = np.array(list(map(len, X))) < 200
train_seq = X[length][:15000]
bio_trans = BioTransformers("esm1_t6_43M_UR50S", device="cuda")

bio_trans.train_masked(
    train_seq,
    lr=1.0e-5,
    warmup_init_lr=1e-7,
    toks_per_batch=2000,
    epochs=20,
    batch_size=16,
    acc_batch_size=256,
    warmup_updates=1024,
    accelerator="ddp",
    checkpoint=None,
    save_last_checkpoint=False,
)
```

## Example : evaluation script

You can easily assees the quality of your finetuning by using the provided function such as `compute_accuracy`.

```python
import biodatasets
import numpy as np
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train sequence with length less than 200 AA
# Test on sequence that was not used for training.
length = np.array(list(map(len, X))) < 200
train_seq = X[length][15000:20000]

bio_trans = BioTransformers("esm1_t6_43M_UR50S", device="cuda", multi_gpu=True)
acc_before = bio_trans.compute_accuracy(train_seq, batch_size=32)
print(f"Accuracy before finetuning : {acc_before}")
```
>> Accuracy before finetuning : 0.46

```python
bio_trans.load_model("logs/finetune_masked/version_X/esm1_t6_43M_UR50S_finetuned.pt")
acc_after = bio_trans.compute_accuracy(train_seq, batch_size=32)
print(f"Accuracy after finetuning : {acc_after}")
```

>> Accuracy after finetuning : 0.76
