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

# 3. Cargar modelo ONNX
print(f"📁 Cargando modelo desde: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("❌ Modelo no encontrado")
    session = None
else:
    try:
        # Configurar opciones para evitar problemas de ejecución
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Crear sesión con CPUExecutionProvider
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
        print("✅ Modelo ONNX cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        session = None

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

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if session is None:
        return JSONResponse({
            "status": "error",
            "mensaje": "Modelo no cargado"
        }, status_code=500)
    
    try:
        imagen_bytes = await file.read()
        img_array = preprocesar_imagen(imagen_bytes)
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)
        prediccion = outputs[0][0]
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
        "modelo_cargado": session is not None,
        "clases": CLASSES
    }

@app.get("/")
async def root():
    return {
        "mensaje": "API Monsite - Clasificador de Cabello",
        "endpoints": ["/", "/health", "/predecir"]
    }
