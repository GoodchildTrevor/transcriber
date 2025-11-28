FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

RUN curl -LO "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp313-cp313-linux_x86_64.whl" && \
    curl -LO "https://download.pytorch.org/whl/cpu/torchaudio-2.5.1%2Bcpu-cp313-cp313-linux_x86_64.whl" && \
    uv pip install --system --no-cache-dir \
        torch-2.5.1+cpu-cp313-cp313-linux_x86_64.whl \
    rm *.whl

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt 

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
