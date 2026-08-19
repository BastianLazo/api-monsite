# ============================================================================
# API MONSITE - CLASIFICACIÓN DE CABELLO CON TENSORFLOW
# ============================================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import io
import os

# 1. Crear app
app = FastAPI(title="Monsite IA - Clasificador de Cabello")

# 2. Configuración
MODEL_PATH = "hair_type_classifier.keras"
CLASSES = ['curly', 'dreadlocks', 'kinky', 'straight', 'wavy']

# 3. Cargar modelo TensorFlow
print(f"📁 Cargando modelo desde: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("❌ Modelo no encontrado")
    model = None
else:
    try:
        model = load_model(MODEL_PATH)
        print("✅ Modelo TensorFlow cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        model = None

def preprocesar_imagen(imagen_bytes):
    # Convertir bytes a imagen
    img = load_img(io.BytesIO(imagen_bytes), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({
            "status": "error",
            "mensaje": "Modelo no cargado"
        }, status_code=500)
    
    try:
        imagen_bytes = await file.read()
        img_array = preprocesar_imagen(imagen_bytes)
        prediccion = model.predict(img_array, verbose=0)
        prediccion = prediccion[0]
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

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "modelo_cargado": model is not None,
        "clases": CLASSES
    }

@app.get("/")
async def root():
    return {
        "mensaje": "API Monsite - Clasificador de Cabello",
        "endpoints": ["/", "/health", "/predecir"]
    }
