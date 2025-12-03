FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

COPY resources/certs/eurocement_root_ca.crt /usr/local/share/ca-certificates/
COPY resources/certs/eurocement_issuing_subca.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    wget \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY resources/pip.conf /etc/pip.conf

RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128   \
    --trusted-host download.pytorch.org \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 

COPY transcriber/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcriber/ .
COPY transcriber/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
