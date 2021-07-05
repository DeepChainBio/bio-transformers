FROM nvidia/cuda:11.1-runtime-ubuntu18.04

ENV CONDA_DIR=/opt/conda
ENV CONDA_PYTHON_VERSION=3
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 liblapack3 openmpi-bin openmpi-common jq git wget gcc libmpich-dev unzip bzip2 build-essential ca-certificates uuid-runtime libxrender1 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

RUN git clone https://github.com/DeepChainBio/bio-transformers

WORKDIR /app/bio-transformers

RUN conda env create -f environment_dev.yaml

SHELL ["/bin/bash", "-c"]

RUN source activate bio-transformers-dev && pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "source activate bio-transformers-dev" >> /root/.bashrc
RUN ${CONDA_DIR}/envs/bio-transformers-dev/bin/pip install -e .

WORKDIR /app
