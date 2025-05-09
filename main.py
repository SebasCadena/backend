from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # ¡Nueva importación!
import cv2
import numpy as np
from ultralytics import YOLO
import io
from fastapi.responses import JSONResponse

app = FastAPI()

# Configura CORS (¡Añade esto antes de tus rutas!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=True
)

# Carga el modelo YOLO (ajusta la ruta)
#modelo = YOLO(r"C:\Users\juans\Documents\CodigosGrado\segmentacion_yoda\runs\segment\train16\weights\best.pt")
modelo = YOLO("best.pt")


# Ruta raíz obligatoria
@app.get("/")
def home():
    return JSONResponse(content={"status": "API activa"}, status_code=200)

@app.post("/segment-leaf")
async def segment_leaf(file: UploadFile = File(...)):
    # 1. Leer la imagen del request
    contents = await file.read()
    imagen = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # 2. Realizar predicción con YOLO
    resultados = modelo.predict(imagen, imgsz=640)

    # 3. Procesar máscaras (código adaptado del tuyo)
    if hasattr(resultados[0], "masks") and resultados[0].masks is not None:
        mascaras = resultados[0].masks.data.cpu().numpy()

        # Encontrar máscara más grande
        mascara_mas_grande = None
        area_maxima = 0

        for mascara in mascaras:
            mascara_uint8 = (mascara * 255).astype("uint8")
            contornos, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > area_maxima:
                    area_maxima = area
                    mascara_mas_grande = mascara_uint8

        if mascara_mas_grande is not None:
            # Aplicar máscara a la imagen original
            mascara_redimensionada = cv2.resize(mascara_mas_grande, (imagen.shape[1], imagen.shape[0]))
            overlay = np.zeros_like(imagen, dtype=np.uint8)
            overlay[np.where(mascara_redimensionada > 0)] = (0, 255, 0)  # Verde
            imagen_segmentada = cv2.addWeighted(imagen, 0.5, overlay, 0.5, 0)

            # Convertir a bytes para la respuesta
            _, img_encoded = cv2.imencode(".png", imagen_segmentada)
            return StreamingResponse(io.BytesIO(img_encoded), media_type="image/png")

    # Si no hay máscaras, devolver la imagen original
    _, img_encoded = cv2.imencode(".png", imagen)
    return StreamingResponse(io.BytesIO(img_encoded), media_type="image/png")