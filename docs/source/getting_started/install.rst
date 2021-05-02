Installation
============

Bio-transformers can be installed in Python 3.7 and external python dependencies are mainly defined in `requirements`_.
DeepReg primarily supports and is regularly tested with Ubuntu and Mac OS.

There are multiple different methods to install Bio-transformers:

1. Clone `Bio-transformers`_ and create a virtual environment using `Anaconda`_ / `Miniconda`_ (**recommended**).
2. Clone `Bio-transformers`_ and build a docker image using the provided docker file. (**not implemented**)
3. Install directly from PyPI release without cloning `Buio-transformers`_.


Install via Conda
-----------------
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
