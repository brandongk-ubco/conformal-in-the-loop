ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.4
ARG CUDA_VERSION=12.1.0
ARG LIGHTNING_VERSION=2.4.0

FROM pytorchlightning/pytorch_lightning:${LIGHTNING_VERSION}-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}


RUN apt-get clean

RUN python -m pip install --upgrade pip

RUN apt update && apt install -y parallel

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PATH_DATASETS=/workspaces/conformal-in-the-loop/datasets/
