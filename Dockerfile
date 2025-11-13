FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Más seguro (parches → menos CVE)
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg && \
    apt-get clean

# Copiar dependencias
COPY requirements.txt .

# Instalarlas
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODO tu proyecto
COPY . .

# Puerto que fly.io expone
ENV PORT=8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
