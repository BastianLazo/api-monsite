# ============================================================================
# API MONSITE - CLASIFICACIÓN DE CABELLO CON ONNX
# ============================================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os

# 1. Crear app
app = FastAPI(title="Monsite IA - Clasificador de Cabello")

# 2. Configuración
MODEL_PATH = "hair_type_classifier.onnx"
CLASSES = ['curly', 'dreadlocks', 'kinky', 'straight', 'wavy']

# 3. Cargar modelo ONNX (al iniciar el servidor)
print(f"📁 Cargando modelo desde: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("❌ Modelo no encontrado. Verifica que esté en la raíz del proyecto")
    session = None
else:
    session = ort.InferenceSession(MODEL_PATH)
    print("✅ Modelo ONNX cargado correctamente")

# 4. Función de preprocesamiento
def preprocesar_imagen(imagen_bytes):
    # Abrir imagen
    img = Image.open(io.BytesIO(imagen_bytes))
    # Redimensionar
    img = img.resize((224, 224))
    # Convertir a array
    img_array = np.array(img).astype(np.float32)
    # Normalización ResNet50
    img_array = img_array / 255.0
    img_array = (img_array - 0.5) * 2
    # Cambiar formato (batch, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. Endpoint de predicción
@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if session is None:
        return JSONResponse({
            "status": "error",
            "mensaje": "Modelo no cargado"
        }, status_code=500)
    
    try:
        # Leer imagen
        imagen_bytes = await file.read()
        
        # Preprocesar
        img_array = preprocesar_imagen(imagen_bytes)
        
        # Ejecutar modelo ONNX
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)
        prediccion = outputs[0][0]
        
        # Interpretar resultado
        clase_idx = int(np.argmax(prediccion))
        clase_nombre = CLASSES[clase_idx]
        confianza = float(np.max(prediccion))
        
        return JSONResponse({
            "status": "success",
            "resultado": {
                "tipo_cabello": clase_nombre,
                "confianza": confianza,
                "probabilidades": {
                    CLASSES[i]: float(prediccion[i]) 
                    for i in range(len(CLASSES))
                }
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "mensaje": str(e)
        }, status_code=400)

# 6. Endpoint de salud
@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "modelo_cargado": session is not None,
        "clases": CLASSES
    }

# 7. Endpoint raíz
@app.get("/")
async def root():
    return {
        "mensaje": "API Monsite - Clasificador de Cabello",
        "endpoints": [
            {"path": "/", "metodo": "GET", "descripcion": "Información de la API"},
            {"path": "/health", "metodo": "GET", "descripcion": "Verificar estado"},
            {"path": "/predecir", "metodo": "POST", "descripcion": "Subir imagen para clasificar"}
        ]
    }
