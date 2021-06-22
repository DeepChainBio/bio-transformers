============
Installation
============

Bio-transformers can be installed in Python 3.7 and external python dependencies are mainly defined in `requirements`_.
There are multiple different methods to install Bio-transformers:

1. Clone `Bio-transformers`_ and create a virtual environment using `Anaconda`_ / `Miniconda`_ (**recommended**).
2. Clone `Bio-transformers`_ and build a docker image using the provided docker file. (**not implemented**)
3. Install directly from PyPI release without cloning `Bio-transformers`_.



Install torch/cuda
------------------

.. WARNING:: ``Bio-transformers`` doesn't manage the installation of cuda toolkit and torch gpu version.

If you want to find a specific version or torch based on your CUDA setup, please refer to this `page <https://pytorch.org/get-started/previous-versions/>`_.

The Dockerfile provided in the `<github repository https://github.com/DeepChainBio/bio-transformers>`_ relies on :
     - pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

Install in conda environment
----------------------------
The recommended method is to install Bio-transformers in a dedicated virtual
environment using `Anaconda`_ / `Miniconda`_.


.. code:: bash

    conda create --name bio-transformers python=3.7 -y
    conda activate bio-transformers
    pip install bio-transformers

.. _Quick Start: quick_start.html
.. _Anaconda: https://docs.anaconda.com/anaconda/install
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Bio-transformers: https://github.com/DeepChainBio/bio-transformers
.. _requirements: https://github.com/DeepChainBio/bio-transformers/blob/main/requirements.txt
