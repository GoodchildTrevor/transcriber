FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --trusted-host download.pytorch.org \
    torch==2.8.0+cu121 \
    torchaudio==2.8.0+cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
