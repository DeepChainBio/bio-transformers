# Change log

# [0.0.7] - 2021-05-12
Note on the release

Features:
 - Add fasta files support for each comupte function.
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
