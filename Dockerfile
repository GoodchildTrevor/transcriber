FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update -oAcquire::AllowInsecureRepositories=true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        ca-certificates software-properties-common wget gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get purge -y --auto-remove \
        --allow-change-held-packages \ 
        libcudnn* libcublas* libcufft* libcurand* libcusolver* libcusparse* \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --trusted-host download.pytorch.org \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 \
    nvidia-cudnn-cu12==9.6.0.74

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
