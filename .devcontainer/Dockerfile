ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.2
ARG CUDA_VERSION=12.1.0
ARG LIGHTNING_VERSION=2.3.3

FROM pytorchlightning/pytorch_lightning:${LIGHTNING_VERSION}-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt update && apt install -y parallel

ENV PATH_DATASETS=/workspaces/conformal-in-the-loop/datasets/
