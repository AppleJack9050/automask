FROM python:3.10-slim
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

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
RUN python3 -m pip install -r requirements.txt

RUN if [ -d "/image-processor/sam3" ]; then \
        cd /image-processor/sam3 && python3 -m pip install -e .; \
    else \
        mkdir -p /image-processor/Grounded-SAM-2/sam2/checkpoints; \
        cd /image-processor/Grounded-SAM-2/sam2/checkpoints && bash /image-processor/Grounded-SAM-2/checkpoints/download_ckpts.sh; \
        cd /image-processor/Grounded-SAM-2 && python3 -m pip install .; \
        cd /image-processor; \
        python3 -m pip install -r Grounded-SAM-2/grounding_dino/requirements.txt; \
        python3 -m pip install Grounded-SAM-2/grounding_dino; \
        cd /image-processor/Grounded-SAM-2/gdino_checkpoints && bash /image-processor/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh; \
    fi

ENV PYTHONPATH=/image-processor/Grounded-SAM-2:$PYTHONPATH

RUN mkdir -p /uploads /processed /saved
RUN chown -R apiuser:apiuser . /uploads /processed /saved
USER apiuser

EXPOSE 80/tcp
CMD ["python3", "-u", "consumer.py"]
