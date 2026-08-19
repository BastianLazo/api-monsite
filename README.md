# API Monsite - Clasificador de Cabello

API para clasificar tipos de cabello usando un modelo ONNX (ResNet50 con fine-tuning).

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información de la API |
| GET | `/health` | Verificar estado del servidor y modelo |
| POST | `/predecir` | Subir imagen para clasificar |

## Uso con Python

```python
import requests

# Probar con una imagen
with open("tu_imagen.jpg", "rb") as f:
    response = requests.post(
        "https://tu-api.onrender.com/predecir",
        files={"file": f}
    )
    print(response.json())
