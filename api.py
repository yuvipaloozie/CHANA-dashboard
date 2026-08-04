import os
import io
import cv2
import base64
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import onnxruntime as ort

from chana.config import IMG_SIZE
from chana.data import pad_image, get_tile_coordinates
from chana.preprocessor import preprocess_v9
from chana.onnx_engine import get_model, predict_batch
from chana.biology import extract_biology

import threading
import webbrowser
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    yield

app = FastAPI(title="CHANA Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_dict = {}

def get_model(name):
    if name not in models_dict:
        path = os.path.join(os.path.dirname(__file__), 'Model Weights', f'{name}.onnx')
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(__file__), 'Model Weights', f'{name}_quantized.onnx')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model {name} not found")
        models_dict[name] = ort.InferenceSession(path)
    return models_dict[name]

@app.post("/preview")
async def preview_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"preview": None}
        
    h, w = img_bgr.shape[:2]
    ratio = 800 / w
    new_size = (800, int(h * ratio))
    resized = cv2.resize(img_bgr, new_size)
    
    _, buffer = cv2.imencode('.jpg', resized)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return {"preview": f"data:image/jpeg;base64,{b64_img}"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), model_type: str = Form("unetplusplus")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def generate():
        session = get_model(model_type)
        padded_img, h, w = pad_image(img_bgr, IMG_SIZE)
        coords = get_tile_coordinates(padded_img.shape, IMG_SIZE)
        
        total_cells = 0
        all_areas = []
        all_circs = []
        all_solids = []
        
        heatmap_data = np.zeros(padded_img.shape[:2], dtype=np.float32)

        total_coords = len(coords)
        for i, (y, x) in enumerate(coords):
            patch = padded_img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            preprocessed_patch = preprocess_v9(patch)
            
            # Predict
            batch_arr = np.expand_dims(preprocessed_patch, axis=0).astype(np.float32)
            preds = predict_batch(session, batch_arr)
            
            # Handle deep supervision models (multiple outputs)
            prob_map = preds[-1] if len(preds) > 1 else preds[0]
            
            # Usually [H, W, 1] or [1, H, W, 1]
            if prob_map.ndim == 4: prob_map = prob_map[0]
            if prob_map.shape[-1] == 1: prob_map = prob_map[..., 0]
            
            mask, count, centroids, areas, circs, solids, eccs = extract_biology(prob_map)
            
            total_cells += count
            all_areas.extend(areas)
            all_circs.extend(circs)
            all_solids.extend(solids)
            
            heatmap_data[y:y+IMG_SIZE, x:x+IMG_SIZE] = count
            
            tile_data = None
            if count > 0:
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                overlay = np.zeros_like(patch_rgb)
                overlay[mask > 0] = [255, 0, 255]
                blended = cv2.addWeighted(patch_rgb, 0.7, overlay, 0.5, 0)
                
                # Confidence Heatmap (Saliency)
                conf_heatmap = cv2.applyColorMap(np.uint8(255 * prob_map), cv2.COLORMAP_TURBO)
                conf_blended = cv2.addWeighted(patch_rgb, 0.4, conf_heatmap, 0.6, 0)
                
                cells = []
                for idx in range(count):
                    cells.append({"id": idx+1, "area": areas[idx], "circularity": circs[idx]})
                    
                    # Draw cell ID at its centroid (centroids are y,x from skimage)
                    cy, cx = centroids[idx]
                    cv2.putText(blended, str(idx+1), (int(cx)-4, int(cy)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(conf_blended, str(idx+1), (int(cx)-4, int(cy)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                b64_img = base64.b64encode(buffer).decode('utf-8')
                
                _, c_buffer = cv2.imencode('.jpg', conf_blended)
                b64_conf = base64.b64encode(c_buffer).decode('utf-8')
                    
                tile_data = {
                    "tile_name": f"tile_y{y}_x{x}",
                    "y": y,
                    "x": x,
                    "count": count,
                    "mask": f"data:image/jpeg;base64,{b64_img}",
                    "confidence": f"data:image/jpeg;base64,{b64_conf}",
                    "cells": cells
                }
            
            progress = (i + 1) / total_coords
            yield json.dumps({"type": "progress", "progress": progress, "tile": tile_data}) + "\n"
            
        heatmap_b64 = None
        max_count = np.max(heatmap_data)
        if max_count > 0:
            heatmap_norm_u8 = ((heatmap_data / max_count) * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm_u8, cv2.COLORMAP_MAGMA)
            alpha = np.expand_dims((heatmap_data / max_count).astype(np.float32), axis=-1)
            blended_heatmap = (padded_img.astype(np.float32) * (1.0 - alpha * 0.7) + heatmap_colored.astype(np.float32) * (alpha * 0.7)).astype(np.uint8)
            _, h_buffer = cv2.imencode('.jpg', blended_heatmap)
            heatmap_b64 = f"data:image/jpeg;base64,{base64.b64encode(h_buffer).decode('utf-8')}"

        avg_size = sum(all_areas) / len(all_areas) if all_areas else 0
        avg_circ = sum(all_circs) / len(all_circs) if all_circs else 0
        avg_solid = sum(all_solids) / len(all_solids) if all_solids else 0

        yield json.dumps({
            "type": "final",
            "total_cells": total_cells,
            "average_size": avg_size,
            "average_circularity": avg_circ,
            "average_solidity": avg_solid,
            "areas": all_areas,
            "circularities": all_circs,
            "heatmap": heatmap_b64
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

frontend_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
if os.path.isdir(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")
    
    @app.get("/")
    @app.get("/{catchall:path}")
    async def serve_react_app(catchall: str = ""):
        return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000)
