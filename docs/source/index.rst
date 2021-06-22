
Bio-transformers : Documentation and Tutorial
=============================================

.. Caution:: Bio-transformers introduces breaking changes replacing ``device`` and ``multi_gpu`` arguments by ``num_gpus``. Multi-gpu inference is now managed with ``ray``, which leverage the full computational capacity of each GPU in contrast to ``torch.DataParallel``

bio-transformers is a python wrapper on top of the ESM/Protbert model,
which are Transformers protein language model, trained on millions on proteins and used to predict embeddings.
This package provide other functionalities that you can use to build apps thanks to `deepchain-apps <https://deepchain-apps.readthedocs.io/en/latest/index.html>`_

Features
--------

.. Note:: Bio-transformers now use `Ray <https://docs.ray.io/en/master/?badge=master#>`_ to manage multi-gpu inference.

Bio-transformers extends and simplifies workflows for manipulating amino acids sequences with Pytorch, and can be
used to test severals pre-trained transformers models without taking into account the synthax of different models.

The main features are:
   - ``compute_loglikelihood``
   - ``compute_probabilities``
   - ``compute_embeddings``
   - ``compute_accuracy``
   - ``finetune``

Our development and all related work involved in the project is public,
and released under the Apache 2.0 license.


Contributors
------------

Bio-transformers is a package belonging to the DeepChainBio repository, maintained by a team of
developers and researchers at Instadeep.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   getting_started/install
   getting_started/quick_start

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Documentation

   documentation/course

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Tutorial

   tutorial/loglikelihood
   tutorial/embeddings
   tutorial/finetuning

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Api reference

    api/biotransformers

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Contributing

   contributing/CHANGELOG
   contributing/CONTRIBUTING

.. _documentation: documentation/course.html
