FROM python:3.11-slim-bookworm

# Desactiva interacción
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Dependencias mínimas del sistema (OpenCV + ffmpeg light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Copiar requirements
COPY requirements.txt .

# 3. Instalar torch CPU PEQUEÑO antes que Ultralytics
# Evita que Ultralytics instale una versión mucho más pesada
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

# 4. Instalar dependencias restantes
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar tu código
COPY . .

# Fly.io usa este puerto
ENV PORT=8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
