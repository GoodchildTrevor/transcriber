FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

RUN pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp313-cp313-linux_x86_64.whl" \
    "https://download.pytorch.org/whl/cpu/torchaudio-2.5.1%2Bcpu-cp313-cp313-linux_x86_64.whl" \

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt 

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
