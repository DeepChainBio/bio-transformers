# Multi-gpus

```{note}
These changes have been introduced in ``bio-transformers`` v0.0.11.
```

The use ``torch.nn.DataParallel`` is strongly [discourage](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead), as a consequence, ``bio-transformers`` relies on [ray](https://docs.ray.io/en/master/?badge=master#) to distribute the compute on multiple GPUs. This parallelization scale far better, with performance increasing with the number of GPUs.

Ray is used only when the ``num_gpus>1``. See the difference below:

```{important}
Note that ray parallelization is only used for inference function. `finetune` method uses pytorch-lightning with its built-in function to train a model.
```

```python
from biotransformers import BioTransformers
import ray

ray.init()

sequences = [...]
bio_trans = BioTransformers("esm1b_t33_650M_UR50S",num_gpus=4)
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=("cls","mean"), batch_size=8)
```

```{note}
You don't have to use ``ray.init()`` when num_gpus=1
```

## Configure GPU environment variable

Sometimes it can be useful to specify which GPU you want to use. It can be done in the terminal or at the beginning of the script. You just have to export the GPU index you want to use.

For example, if you have 8 GPUs but you just want to use 3 of them (0,5,6):

```bash
export CUDA_VISIBLE_DEVICES="0,5,6"
```

or

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,5,6"
```
