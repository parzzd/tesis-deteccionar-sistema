FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU liviano
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1 torchvision==0.17.1

# Copiar requirements
COPY requirements.txt .

# Instalar requirements sin CUDA ni extras
RUN pip install --no-cache-dir -r requirements.txt --no-build-isolation

COPY . .

ENV PORT=8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
