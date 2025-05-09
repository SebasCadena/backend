from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
import logging
from typing import Optional

app = FastAPI()

# Configuración avanzada de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite temporalmente todos los orígenes
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600
)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga del modelo con verificación
try:
    logger.info("⏳ Cargando modelo YOLO...")
    modelo = YOLO("best.pt", device="cpu")  # Fuerza uso de CPU
    logger.info("✅ Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {str(e)}")
    raise RuntimeError("No se pudo cargar el modelo YOLO") from e


@app.get("/")
def health_check():
    return JSONResponse(
        content={
            "status": "API activa",
            "modelo": "cargado" if modelo else "no cargado"
        },
        status_code=200
    )


@app.post("/segment-leaf")
async def segment_leaf(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Recibiendo archivo: {file.filename}")

        # Verificación básica del archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Solo se permiten imágenes")

        contents = await file.read()
        logger.info(f"📏 Tamaño de imagen recibida: {len(contents) / 1024:.2f} KB")

        # Decodificación robusta de la imagen
        nparr = np.frombuffer(contents, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen is None:
            logger.error("⚠️ No se pudo decodificar la imagen")
            raise HTTPException(400, "Formato de imagen no soportado")

        logger.info(f"🖼️ Dimensión de imagen: {imagen.shape}")

        # Procesamiento con YOLO
        logger.info("🔍 Ejecutando predicción...")
        resultados = modelo.predict(
            imagen,
            imgsz=640,
            conf=0.25,  # Ajusta según necesites
            device="cpu"
        )
        logger.info("🎯 Predicción completada")

        # Procesamiento de máscaras
        if not hasattr(resultados[0], "masks") or resultados[0].masks is None:
            logger.warning("⚠️ No se detectaron máscaras")
            raise HTTPException(400, "No se detectaron objetos en la imagen")

        mascaras = resultados[0].masks.data.cpu().numpy()
        logger.info(f"🔄 Procesando {len(mascaras)} máscaras...")

        # Encontrar máscara más grande
        mascara_mas_grande = max(
            ((m * 255).astype("uint8") for m in mascaras),
            key=lambda m: cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].size,
            default=None
        )

        if mascara_mas_grande is None:
            logger.warning("⚠️ Máscara más grande no encontrada")
            raise HTTPException(400, "No se pudo segmentar la imagen")

        # Aplicar máscara
        mascara_rsz = cv2.resize(mascara_mas_grande, (imagen.shape[1], imagen.shape[0]))
        overlay = np.zeros_like(imagen)
        overlay[np.where(mascara_rsz > 0)] = (0, 255, 0)  # Verde
        imagen_segmentada = cv2.addWeighted(imagen, 0.7, overlay, 0.3, 0)

        # Codificar respuesta
        _, img_encoded = cv2.imencode(".png", imagen_segmentada)
        logger.info("✅ Procesamiento completado")

        return StreamingResponse(
            io.BytesIO(img_encoded),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=segmentada.png"}
        )

    except HTTPException:
        raise  # Re-lanza las excepciones HTTP que ya manejamos
    except Exception as e:
        logger.error(f"🔥 Error crítico: {str(e)}", exc_info=True)
        raise HTTPException(500, "Error interno procesando la imagen") from e