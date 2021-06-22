# Embeddings

The library allow to easily compute embeddings with a specific model in the backend.

.. code-block:: python

    from biotransformers import BioTransformers

    sequences = [...]
    bio_trans = BioTransformers("esm1b_t33_650M_UR50S",num_gpus=1)
    embeddings = bio_trans.compute_embeddings(sequences, pool_mode=("cls","mean"), batch_size=8)

By default, the ``pool_mode`` argument contains 3 mode:

    - ``cls`` : return the `<CLS>` token embedding in the sequence.
    - ``mean`` : if sequence has shape (num_token, embedding_size), the num_token dimension is averaging and the embedding has shape (num_token,)
    - ``full`` : no pooling function applied, all the embeddings for each sequence are return.


## Multi-gpu inference

If you want to make the inference on several GPUs, you have to intialize ray as below to use instantiate multiple workers.

```{tip}
``batch_size`` corresponds to the number of sequence that you want to distribute on each GPU.
```

.. code-block:: python

    from biotransformers import BioTransformers
    import ray

    ray.init()
    sequences = [...]
    bio_trans = BioTransformers("esm1b_t33_650M_UR50S",num_gpus=4)
    embeddings = bio_trans.compute_embeddings(sequences, pool_mode=("cls","mean"), batch_size=8)
