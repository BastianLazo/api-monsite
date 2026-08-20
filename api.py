# ============================================================================
# API MONSITE - CLASIFICACIÓN DE CABELLO
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
import gdown

# 1. Crear app
app = FastAPI(title="Monsite IA - Clasificador de Cabello")

# 2. Configuración
# --- !!! REEMPLAZA ESTE ID CON EL DE TU ARCHIVO !!! ---
FILE_ID = "1JQodxYXrihSSGfOA8vwfSpbXqsmsjf9x"  # <--- ¡YA ESTÁ ACTUALIZADO!
# --- ------------------------------------------------- ---
MODEL_FILENAME = "hair_type_classifier.keras"
MODEL_PATH = MODEL_FILENAME
CLASSES = ['curly', 'dreadlocks', 'kinky', 'straight', 'wavy']

# 3. Descargar y Cargar Modelo
print(f"📁 Buscando modelo en: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("📥 Modelo no encontrado. Descargando desde Google Drive...")
    try:
        # Construir la URL de descarga directa
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Modelo descargado correctamente.")
    except Exception as e:
        print(f"❌ Error al descargar el modelo: {e}")
        model = None
        # Salir o manejar el error, ya que sin modelo la API no puede funcionar
        # exit(1) # Podrías descomentar esto para forzar un fallo en el inicio.

# Cargar el modelo (ahora debería existir localmente)
try:
    # Desactivar warnings de compilación para una salida más limpia
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Cargar el modelo sin compilarlo (solo para inferencia)
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Modelo TensorFlow cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    model = None

# (El resto de tu código de preprocesamiento y endpoints se mantiene igual)
def preprocesar_imagen(imagen_bytes):
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
