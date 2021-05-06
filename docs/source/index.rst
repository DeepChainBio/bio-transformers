
Bio-transformers : Documentation and Tutorial
=============================================

bio-transformers is a python wrapper on top of the ESM/Protbert model,
which are Transformers protein language model, trained on millions on proteins and used to predict embeddings.
This package provide other functionalities like compute the loglikelihood of a protein or the accuracy of a model.

Features
--------

Bio-transformers extends and simplifies workflows for manipulating amino acids sequences with Pytorch, and can be
used to test severals pre-trained transformers models without taking into account the synthax of different models.

Our development and all related work involved in the project is public,
and released under the Apache 2.0 license.

Where to start
--------------
If you want to know more about transformers ans biology, refer to this _`documentation`


Contributors
------------

Bip-transformers is a package belonging to the DeepChainBio repository, maintained by a team of
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

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Api reference

    api/biotransformers


.. _documentation: documentation/course.html
