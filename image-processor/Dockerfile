FROM python:3.10-slim

WORKDIR /image-processor
COPY . /image-processor

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc g++ \
    cmake \
    ninja-build \
    git \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r apiuser && useradd -r -g apiuser apiuser -m

RUN mkdir -p /image-processor/Grounded-SAM-2/sam2/checkpoints
RUN cd /image-processor/Grounded-SAM-2/sam2/checkpoints && bash /image-processor/Grounded-SAM-2/checkpoints/download_ckpts.sh
RUN cd /image-processor/Grounded-SAM-2 && python3 -m pip install .

RUN python3 -m pip install -r Grounded-SAM-2/grounding_dino/requirements.txt
RUN python3 -m pip install Grounded-SAM-2/grounding_dino
RUN cd /image-processor/Grounded-SAM-2/gdino_checkpoints && bash /image-processor/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh

ENV PYTHONPATH=/image-processor/Grounded-SAM-2:$PYTHONPATH

RUN python3 -m pip install -r requirements.txt
RUN mkdir -p /uploads /processed /saved
RUN chown -R apiuser:apiuser . /uploads /processed /saved
USER apiuser

EXPOSE 80/tcp
CMD ["python3", "-u", "consumer.py"]
