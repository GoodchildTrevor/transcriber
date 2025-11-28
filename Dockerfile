FROM python:3.13-slim

WORKDIR /app

COPY resources/certs/eurocement_root_ca.crt /usr/local/share/ca-certificates/
COPY resources/certs/eurocement_issuing_subca.crt /usr/local/share/ca-certificates/

RUN update-ca-certificates

COPY resources/pip.conf /etc/pip.conf

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --trusted-host download.pytorch.org \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    xformers==0.0.33

RUN pip install --no-cache-dir uv

COPY transcriber/requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt 

COPY transcriber/ .

COPY transcriber/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]