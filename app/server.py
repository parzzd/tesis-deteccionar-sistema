# app/server.py
# uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

import os, io, cv2, time, json, base64, asyncio, collections
from typing import Dict, Any, Optional, Set, List, Tuple
from pathlib import Path

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# =========================
# Configuración
# =========================
ROOT_DIR     = Path(__file__).resolve().parent
STATIC_DIR   = (ROOT_DIR / "static").resolve()

# Modelos (ajusta nombres si cambiaste)
KERAS_MODEL  = Path("./models_mix/mix_cnn_lstm_T32_F51.keras")
NORM_STATS   = Path("./models_mix/mix_cnn_lstm_T32_F51_norm_stats.npz")
THRESH_JSON  = Path("./models_mix/mix_cnn_lstm_T32_F51_threshold.json")  # {"best_threshold": 0.xx}
LGBM_PKL     = Path("./models_mix/lgbm_model.pkl")  # opcional; si no existe, fusión se apaga

# YOLO pose
POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolo11m-pose.pt")
IMGSZ        = int(os.environ.get("POSE_IMGSZ", "640"))
CONF_POSE    = float(os.environ.get("POSE_CONF", "0.25"))
IOU_POSE     = float(os.environ.get("POSE_IOU", "0.50"))
TOPK         = int(os.environ.get("POSE_TOPK", "4"))
## http://localhost:8000/
# Ventanas
SEQ_LEN      = 32           # debe coincidir con el modelo cargado
STRIDE       = 1            # salto entre frames para la cola interna
CONF_MIN     = 0.10         # “visible” si conf_joint >= CONF_MIN
MIN_VIS_FRAC = 0.50         # descartar ventana si <50% frames con algo visible

# Fusión con LGBM
FUSION_W     = 0.50         # 0→solo Keras, 1→solo LGBM (si hay LGBM)
POOL_METHOD  = "topk"       # "max" | "mean" | "topk"
TOPK_FRAC    = 0.20         # para pool top-k de scores en el video
HYST_GAP     = 0.10         # histéresis: thr_off = thr_on - HYST_GAP

# =========================
# Utilidades de pose/features
# =========================
def pool_frame_to_51(kps_f: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    kps_f: (K,17,3) con (x,y,conf_joint). Devuelve (51,) normalizado por W,H.
    Por joint, elige la persona con mayor conf en ese joint.
    """
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)

    conf_j = kps_f[..., 2]
    conf_j = np.nan_to_num(conf_j, nan=0.0)
    K = kps_f.shape[0]

    for j in range(17):
        if K == 0: break
        idx = int(np.argmax(conf_j[:, j]))
        c = conf_j[idx, j]
        if c > 0:
            x, y, _ = kps_f[idx, j, :]
            if np.isfinite(x) and np.isfinite(y):
                out[j, 0] = np.clip(x / max(W, 1), 0.0, 1.0)
                out[j, 1] = np.clip(y / max(H, 1), 0.0, 1.0)
                out[j, 2] = float(np.clip(c, 0.0, 1.0))
    return out.reshape(-1)

def frame_visible(kps_f: np.ndarray, conf_min: float = CONF_MIN) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False
    conf = kps_f[..., 2]
    conf = np.nan_to_num(conf, nan=0.0)
    return bool((conf >= conf_min).any())

def pool_scores(scores: List[float], pool: str = "topk", topk_frac: float = 0.2) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if pool == "max":
        return float(arr.max())
    if pool == "mean":
        return float(arr.mean())
    k = max(1, int(len(arr) * topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())

# =========================
# Carga de modelos/artefactos
# =========================
def load_artifacts():
    # Keras
    if not KERAS_MODEL.exists():
        raise FileNotFoundError(f"No existe el modelo Keras: {KERAS_MODEL}")
    keras_model = load_model(str(KERAS_MODEL), compile=False)

    # Stats de normalización
    if not NORM_STATS.exists():
        raise FileNotFoundError(f"No existe norm stats: {NORM_STATS}")
    stats = np.load(NORM_STATS)
    mu = stats["mean"]  # (1,F)
    sd = stats["std"]   # (1,F)

    # Umbral óptimo
    if THRESH_JSON.exists():
        thr = float(json.loads(Path(THRESH_JSON).read_text(encoding="utf-8")).get("best_threshold", 0.5))
    else:
        thr = 0.5

    # LGBM (opcional)
    lgbm = None
    if LGBM_PKL.exists():
        try:
            lgbm = joblib.load(LGBM_PKL)
            print(f"[BOOT] LGBM ON → {LGBM_PKL}")
        except Exception as e:
            print(f"[BOOT] LGBM: fallo al cargar ({e}), continuo sin LGBM")

    # YOLO Pose
    pose = YOLO(POSE_WEIGHTS)

    return keras_model, mu.astype("float32"), sd.astype("float32"), thr, lgbm, pose

KERAS, MU, SD, THR_ON, LGBM, POSE = load_artifacts()
THR_OFF = max(0.0, THR_ON - HYST_GAP)

print(f"[BOOT] Keras={KERAS_MODEL} | THR_ON={THR_ON:.2f} THR_OFF={THR_OFF:.2f} | T={SEQ_LEN} | FusionW={FUSION_W:.2f} | LGBM={'ON' if LGBM is not None else 'OFF'}")

# =========================
# FastAPI + static
# =========================
app = FastAPI(title="Violence Detection – Window Level (Keras + optional LGBM)")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root():
        idx = STATIC_DIR / "index.html"
        return FileResponse(str(idx)) if idx.exists() else JSONResponse({"ok": True, "msg": "sin index.html"})
else:
    @app.get("/")
    def root():
        return JSONResponse({"ok": True, "msg": "Servidor arriba (sin carpeta static)"})


# =========================
# Gestión de cámaras simples
# =========================
class CameraConfig(BaseModel):
    cam_id: str
    src: str  # índice de cam (e.g., "0") o ruta/rtsp

CAMERAS: Dict[str, CameraConfig] = {}

@app.get("/cameras")
def list_cameras():
    return list(CAMERAS.values())

@app.post("/cameras")
def add_camera(cfg: CameraConfig):
    CAMERAS[cfg.cam_id] = cfg
    return {"ok": True}

@app.delete("/cameras/{cam_id}")
def del_camera(cam_id: str):
    CAMERAS.pop(cam_id, None)
    return {"ok": True}


# =========================
# Worker por cámara (sin tracking por persona)
# =========================
class CameraWorker:
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # buffer de ventana para el video (no por persona)
        self.win_feats: collections.deque = collections.deque(maxlen=SEQ_LEN)  # (51,) por frame
        self.win_vis:   collections.deque = collections.deque(maxlen=SEQ_LEN)  # bool por frame
        self.video_scores: List[float] = []  # scores por ventana
        self.on_state = False  # estado ON/OFF por histéresis

        self.W = None
        self.H = None

    async def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.running = False
        if self.task:
            await asyncio.sleep(0)
            self.task.cancel()
            self.task = None

    def _norm_apply(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica normalización por-feature con MU/SD de entrenamiento.
        X: (1, T, F)
        """
        T, F = X.shape[1], X.shape[2]
        X2 = X.reshape(-1, F)
        Xn = (X2 - MU) / (SD + 1e-6)
        return Xn.reshape(1, T, F).astype("float32")

    def _predict_window(self, Xw: np.ndarray) -> float:
        """
        Xw: (T, 51), retorna prob(1) fusionada (Keras [+ LGBM opcional]).
        """
        X = Xw[np.newaxis, ...]                       # (1,T,51)
        X = self._norm_apply(X)                       # normaliza
        p_keras = float(KERAS.predict(X, verbose=0).ravel()[0])

        if LGBM is None or FUSION_W <= 0.0:
            return p_keras

        # Features tabulares para LGBM (estadísticas simples por joint y velocidades aproximadas)
        # -- mismas que se usaron al entrenar el LGBM (si tú lo definiste distinto, ajusta aquí) --
        x = Xw  # (T,51) → reinterpreta a (T,17,3)
        x3 = x.reshape(x.shape[0], 17, 3)
        xy = x3[..., :2]         # (T,17,2)
        dx = np.diff(xy, axis=0, prepend=xy[0:1])   # (T,17,2)
        v  = np.linalg.norm(dx, axis=-1)            # (T,17)

        def stats(a):
            return np.concatenate([a.mean(0).ravel(),
                                   a.std(0).ravel(),
                                   a.min(0).ravel(),
                                   a.max(0).ravel()], axis=0)

        feat = np.concatenate([stats(xy[...,0]), stats(xy[...,1]), stats(v)], axis=0).astype(np.float32)
        p_lgbm = float(LGBM.predict_proba(feat.reshape(1, -1))[:, 1][0])

        return (1.0 - FUSION_W) * p_keras + FUSION_W * p_lgbm

    async def run(self):
        cap = None
        try:
            # abrir fuente
            if self.src.strip().isdigit():
                cap = cv2.VideoCapture(int(self.src.strip()))
            else:
                cap = cv2.VideoCapture(self.src)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                await self.broadcast({"type": "error", "msg": "no-open"})
                return

            fps_smooth, t0 = 0.0, time.time()
            self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    await self.broadcast({"type": "error", "msg": "eof"})
                    break

                # === Pose (sin tracking personas) ===
                res = POSE.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE,
                                   verbose=False, half=False)[0]

                kps_f = None
                if res.keypoints is not None and res.keypoints.xy is not None and res.keypoints.xy.shape[0] > 0:
                    xy = res.keypoints.xy.detach().cpu().numpy()  # (P,17,2)
                    c  = getattr(res.keypoints, "confidence", None) or getattr(res.keypoints, "conf", None)
                    if c is not None:
                        c = c.detach().cpu().numpy()              # (P,17)
                    else:
                        c = np.ones(xy.shape[:2], dtype=np.float32)
                    # ordenar personas por conf media
                    order = np.argsort(-c.mean(axis=1))
                    P = min(len(order), TOPK)
                    xy = xy[order[:P]]
                    c  = c[order[:P]]
                    # ensamblar (x,y,conf_joint)
                    kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)  # (P,17,3)

                # features por frame
                feat51 = pool_frame_to_51(kps_f, self.W, self.H)  # (51,)
                vis = frame_visible(kps_f, CONF_MIN)

                self.win_feats.append(feat51)
                self.win_vis.append(1.0 if vis else 0.0)

                p_win = 0.0
                p_vid = 0.0

                if len(self.win_feats) == SEQ_LEN:
                    vis_frac = np.mean(self.win_vis)
                    if vis_frac >= MIN_VIS_FRAC:
                        Xw = np.stack(self.win_feats, axis=0)  # (T,51)
                        p_win = self._predict_window(Xw)
                        self.video_scores.append(p_win)
                        p_vid = pool_scores(self.video_scores, pool=POOL_METHOD, topk_frac=TOPK_FRAC)

                        # histéresis
                        if not self.on_state and p_vid >= THR_ON:
                            self.on_state = True
                            await self.broadcast({"type": "alert", "cam_id": self.cam_id, "prob": float(p_vid), "ts": time.time()})
                        elif self.on_state and p_vid <= THR_OFF:
                            self.on_state = False

                # HUD
                try:
                    show = res.plot() if res is not None else frame
                except Exception:
                    show = frame

                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(1e-6, time.time() - t0)); t0 = time.time()
                cv2.putText(show, f"fps={fps_smooth:.1f}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)
                cv2.putText(show, f"p_win={p_win:.2f}  p_vid={p_vid:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)
                if self.on_state:
                    cv2.putText(show, "ALERTA", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 3)

                # enviar frame
                _, buf = cv2.imencode(".jpg", show)
                b64 = base64.b64encode(buf.tobytes()).decode()
                payload = {
                    "type": "frame", "cam_id": self.cam_id,
                    "p_win": float(p_win), "p_vid": float(p_vid),
                    "on": bool(self.on_state), "ts": time.time(),
                    "jpg_b64": b64
                }
                await self.broadcast(payload)

                # control de ritmo
                await asyncio.sleep(0.005)
        finally:
            if cap is not None:
                cap.release()

    async def broadcast(self, msg: Dict[str, Any]):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                try: await ws.close()
                except Exception: pass
                self.clients.discard(ws)

WORKERS: Dict[str, CameraWorker] = {}

# =========================
# WebSocket (simple, sin JWT)
# =========================
@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    cfg = CAMERAS.get(cam_id)
    if cfg is None:
        await websocket.send_json({"type": "error", "msg": "cam-not-found"})
        await websocket.close()
        return

    worker = WORKERS.get(cam_id)
    if worker is None:
        worker = CameraWorker(cam_id, cfg.src)
        WORKERS[cam_id] = worker
        await worker.start()

    worker.clients.add(websocket)

    try:
        while True:
            _ = await websocket.receive_text()  # reservado: comandos
    except WebSocketDisconnect:
        worker.clients.discard(websocket)
