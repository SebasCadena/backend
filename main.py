from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
import logging
import torch
from typing import Optional

app = FastAPI()

# Configuración avanzada de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=True,
    max_age=3600
)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga del modelo con verificación
try:
    logger.info("⏳ Cargando modelo YOLO...")
    modelo = YOLO("best.pt", task="segment", type='v8')
    logger.info("✅ Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {str(e)}")
    raise RuntimeError("No se pudo cargar el modelo YOLO") from e


@app.post("/segment-leaf")
async def segment_leaf(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Recibiendo archivo: {file.filename}")

        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Solo se permiten imágenes")

        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen is None:
            logger.error("⚠️ No se pudo decodificar la imagen")
            raise HTTPException(400, "Formato de imagen no soportado")

        logger.info(f"🖼️ Dimensión de imagen: {imagen.shape}")

        logger.info("🔍 Ejecutando predicción...")
        resultados = modelo.predict(
            imagen,
            imgsz=640,
            conf=0.4,
            device="cpu",
            half=False
        )
        logger.info("🎯 Predicción completada")

        if not hasattr(resultados[0], "masks") or resultados[0].masks is None:
            logger.warning("⚠️ No se detectaron máscaras")
            raise HTTPException(400, "No se detectaron objetos en la imagen")

        # Procesar máscaras y generar el archivo .txt con las segmentaciones
        etiquetas_yolo = io.StringIO()
        logger.info(f"📝 Generando etiquetas para {len(resultados[0].masks.xyn)} máscaras...")

        for cls, mask in zip(resultados[0].boxes.cls.cpu().numpy(), resultados[0].masks.xyn):
            cls_int = int(cls)
            puntos = ["{:.6f} {:.6f}".format(coord[0], coord[1]) for coord in mask]
            linea = f"{cls_int} " + " ".join(puntos)
            etiquetas_yolo.write(linea + "\n")

        # Convertir contenido a bytes para respuesta
        etiquetas_yolo.seek(0)
        etiquetas_bytes = io.BytesIO(etiquetas_yolo.getvalue().encode())

        logger.info("✅ Archivo de etiquetas generado correctamente")

        return StreamingResponse(
            etiquetas_bytes,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={file.filename.split('.')[0]}_labels.txt"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Error crítico: {str(e)}", exc_info=True)
        raise HTTPException(500, "Error interno procesando la imagen") from e

@app.get("/")
def health_check():
    return JSONResponse(
        content={
            "status": "API activa",
            "modelo": "cargado" if modelo else "no cargado"
        },
        status_code=200
    )
