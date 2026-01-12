FROM ubuntu:22.04

WORKDIR /api
COPY . /api

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential \
    gcc g++ \
    cmake \
    ninja-build \
    git \ 
    libgl1 \
    libglib2.0-0 \
    libglib2.0-dev \
    wget \
    ca-certificates

RUN groupadd -r apiuser && useradd -r -g apiuser apiuser


RUN mkdir /api/Grounded-SAM-2/sam2/checkpoints
RUN cd /api/Grounded-SAM-2/sam2/checkpoints && bash /api//Grounded-SAM-2/checkpoints/download_ckpts.sh
RUN cd /api/Grounded-SAM-2 && python3 -m pip install -e .

# SET UP Grounding DINO
RUN python3 -m pip install -r Grounded-SAM-2/grounding_dino/requirements.txt
RUN python3 -m pip install -e Grounded-SAM-2/grounding_dino
RUN cd /api/Grounded-SAM-2/gdino_checkpoints && bash /api/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh

ENV PYTHONPATH=/api:/api/Grounded-SAM-2:$PYTHONPATH

# General requirements
RUN python3 -m pip install -r requirements.txt

RUN chown -R apiuser:apiuser .
USER apiuser

EXPOSE 80/tcp
CMD ["uvicorn", "endpoints:app", "--app-dir", "/api", "--host", "0.0.0.0", "--port", "80"]
