FROM python:3.12-slim

RUN apt update && apt install -y ffmpeg

RUN python -m venv /opt/venv

RUN /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        --trusted-host download.pytorch.org \
        torch==2.8.0+cu128 \
        torchaudio==2.8.0+cu128 \
        nvidia-cudnn-cu12==9.10.2.21

COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
COPY /entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/opt/venv/bin:$PATH"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
