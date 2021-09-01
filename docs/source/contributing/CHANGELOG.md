# Change log

# [0.1.8] - 2021-07-29

Features:
  - Add compute_mutation_score method to evaluate a set of mutation on a sequence.
    Metric based on [paper](<https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full.pdf>)

# [0.1.7] - 2021-07-19

Features:
  - Add esm1v_t33_650M_UR90S_1 model.

# [0.1.6] - 2021-07-09

Fixed:
 - Fix filtering of logits which impacts loglikelihood computation
 - Fix fasta file reading in compute_loglikelihood

Features:
  - Add `normalize` mode in compute_loglikelihood.

# [0.1.3] - 2021-07-01

Features:
 - Add msa-transformers for methods:
    - compute_logits
    - compute_embeddings
    - compute_probabilities
    - compute_accuracy

Fixed:
 - Remove torch DataParallel wrapper.

# [0.1.0] - 2021-07-01

Features:
 - Add ray worker for multi-gpus inference

Removed:
 - Remove torch DataParallel wrapper.

# [0.0.10] - 2021-06-14
Note on the release

Features:
 - Add BIO_LOG_LEVEL environnement variable to control logging message (logger)
 - Check if every unique amino acids in sequences are in tokens_list (compute_probabilities)

Fixed:
 - Add shuffling in batch_sampler (lightning_utils)
 - Fix tokens argument for dataloader (lightning_utils)
 - Fix rtd CI to separates docs and package environment.

Changed:
 - Modified the signature of some functions to improve clarity (tansformers_wrappers)
 - Update `train_masked` method to `finetune` (tansformers_wrappers)
 - `compute_embeddings` with option `full` return a list of embeddingsn, no matter the size (tansformers_wrappers)

Removed:
 - Remove the tokens_list argument when not necessary and tried to make its usage clearer (tansformers_wrappers)
 - Remove functions (tansformers_wrappers):
    - _filter_and_pool_embeddings
    - _split_logits
    -  _slabels_remaping
    - _filter_logits
    -  _filter_loglikelihood
    - _compute_accuracy
    - _compute_calibration


# [0.0.9] - 2021-06-04

Fixed:
 - Batch_sampler issue

# [0.0.8] - 2021-06-03
Note on the release

Features:
 - Merge ESM/protbert for finetuning model with pytorch-lightning
 - Possibility to restore a training session.

Fixed:
 - Fix conflicts when saving model with DDP
 - Fix loading checkpoint created by pytorch-lightning


# [0.0.7] - 2021-05-12
Note on the release

Features:
 - Add fasta files support for each compute function.
 - Add train_masked function to finetune model on custom dataset. (Only ESM for the moment, protbert is coming.)

Docs:
 - Update documentation to add tutorial on training.

Changed:
 - GPU is used by default if found, even if not specified.

# [0.0.6] - 2021-05-24
Note on the release

Fixed:
 - Update torch dependencies to be less restrictive. Create conflict with other packages.

# [0.0.5] - 2021-05-12

Note on the release

Added
 - added multi-gpu support for inference
 - added function to finetuned a model on a specific dataset on multi-gpu

Changed

Fixed
