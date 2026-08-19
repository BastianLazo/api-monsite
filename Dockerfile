FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para Pillow y ONNX Runtime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero (para aprovechar caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
