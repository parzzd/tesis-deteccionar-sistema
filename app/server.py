# uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
import os, time, json, base64, asyncio
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
from collections import deque

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
# venv\Scripts\python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

ROOT_DIR   = Path(__file__).resolve().parent
STATIC_DIR = (ROOT_DIR / "static").resolve()

KERAS_MODEL = Path("./models_mix/mix_cnn_lstm_T32_F51.keras")
NORM_STATS  = Path("./models_mix/mix_cnn_lstm_T32_F51_norm_stats.npz")
THRESH_JSON = Path("./models_mix/mix_cnn_lstm_T32_F51_threshold.json")
LGBM_PKL    = Path("./models_mix/lgbm_model.pkl")  # opcional

# Pose
POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolo11m-pose.pt")
IMGSZ        = int(os.environ.get("POSE_IMGSZ", "640"))
CONF_POSE    = float(os.environ.get("POSE_CONF", "0.25"))
IOU_POSE     = float(os.environ.get("POSE_IOU", "0.50"))
TOPK         = int(os.environ.get("POSE_TOPK", "4"))

# Ventanas (debe calzar con el modelo)
SEQ_LEN      = 32
CONF_MIN     = 0.10
MIN_VIS_FRAC = 0.30

# Fusión (Keras + LGBM opcional)
FUSION_W    = 0.50     # 0→solo Keras, 1→solo LGBM
POOL_METHOD = "topk"   # max|mean|topk
TOPK_FRAC   = 0.20
HYST_GAP    = 0.10     # thr_off = thr_on - gap

# Preview (NO afecta inferencia)
SEND_FPS      = float(os.environ.get("SEND_FPS", "10"))
FRAME_WIDTH   = int(os.environ.get("FRAME_WIDTH", "720"))
JPEG_QUALITY  = int(os.environ.get("JPEG_QUALITY", "65"))
DRAW_OVERLAY  = os.getenv("DRAW_OVERLAY", "1") != "0"
VIDEO_MAX_SCORES = int(os.getenv("VIDEO_MAX_SCORES", "900"))  # ~1–2 min

# =========================
# UTILS (pose → 51f, pooling, etc.)
# =========================
def pool_frame_to_51(kps_f: np.ndarray, W: int, H: int) -> np.ndarray:
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)
    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
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
    v = out.reshape(-1)
    return np.nan_to_num(v, nan=0.0)

def frame_visible(kps_f: np.ndarray, conf_min: float = CONF_MIN) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False
    conf = np.nan_to_num(kps_f[..., 2], nan=0.0)
    return bool((conf >= conf_min).any())

def pool_scores(scores: List[float], pool: str = "topk", topk_frac: float = 0.2) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if pool == "max":  return float(arr.max())
    if pool == "mean": return float(arr.mean())
    k = max(1, int(len(arr) * topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())

# =========================
# CARGA MODELOS
# =========================
def load_artifacts():
    if not KERAS_MODEL.exists():
        raise FileNotFoundError(f"No existe Keras: {KERAS_MODEL}")
    keras_model = load_model(str(KERAS_MODEL), compile=False)

    if not NORM_STATS.exists():
        raise FileNotFoundError(f"No existe norm stats: {NORM_STATS}")
    stats = np.load(NORM_STATS)
    mu = stats["mean"].astype("float32")
    sd = stats["std"].astype("float32")

    if THRESH_JSON.exists():
        thr = float(json.loads(Path(THRESH_JSON).read_text(encoding="utf-8")).get("best_threshold", 0.5))
    else:
        thr = 0.5

    lgbm = None
    if LGBM_PKL.exists():
        try:
            lgbm = joblib.load(LGBM_PKL)
            print(f"[BOOT] LGBM ON → {LGBM_PKL}")
        except Exception as e:
            print(f"[BOOT] LGBM: fallo al cargar ({e}), continúo sin LGBM")

    pose = YOLO(POSE_WEIGHTS)
    return keras_model, mu, sd, thr, lgbm, pose

KERAS, MU, SD, THR_ON, LGBM, POSE = load_artifacts()
THR_OFF = max(0.0, THR_ON - HYST_GAP)
print(f"[BOOT] Keras={KERAS_MODEL} | THR_ON={THR_ON:.2f} THR_OFF={THR_OFF:.2f} | T={SEQ_LEN} | FusionW={FUSION_W:.2f} | LGBM={'ON' if LGBM is not None else 'OFF'}")

# =========================
# FASTAPI + STATIC + LOGIN
# =========================
app = FastAPI(title="VigilIA – Detección (Keras + LGBM opcional)")

# servir estáticos
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Home → login.html
@app.get("/")
def home():
    idx = STATIC_DIR / "login.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"ok": True, "msg": "sube app/static/login.html"})

@app.get("/login-page")
def login_page():
    f = STATIC_DIR / "login.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"ok": False, "msg": "no login.html"})

@app.get("/login-fail")
def login_fail():
    f = STATIC_DIR / "login_fail.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"ok": False, "msg": "no login_fail.html"})

@app.get("/dashboard")
def dashboard():
    f = STATIC_DIR / "index.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"ok": False, "msg": "no index.html"})

@app.get("/health")
def health():
    return {"ok": True}

# --- LOGIN SIMPLE (.env) ---
SIC_EMAIL    = os.getenv("SIC_EMAIL", "")
SIC_PASSWORD = os.getenv("SIC_PASSWORD", "")

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    if not SIC_EMAIL or not SIC_PASSWORD:
        raise HTTPException(status_code=500, detail="Credenciales no configuradas en .env")
    if data.email != SIC_EMAIL or data.password != SIC_PASSWORD:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    return TokenResponse(access_token="sicher_dummy_token")

# =========================
# CÁMARAS
# =========================
class CameraConfig(BaseModel):
    cam_id: str
    src: str  # "0" ó ruta/rtsp

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
# WORKER (sin tracking; ventana global)
# =========================
class CameraWorker:
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        self.win_feats: deque = deque(maxlen=SEQ_LEN)     # (51,)
        self.win_vis:   deque = deque(maxlen=SEQ_LEN)     # bool
        self.video_scores: deque = deque(maxlen=VIDEO_MAX_SCORES)
        self.on_state = False

        self.W, self.H = None, None
        self._last_send_ts = 0.0  # control preview

    async def start(self):
        if self.running: return
        self.running = True
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.running = False
        if self.task:
            await asyncio.sleep(0)
            self.task.cancel()
            self.task = None

    def _norm_apply(self, X: np.ndarray) -> np.ndarray:
        T, F = X.shape[1], X.shape[2]
        X2 = X.reshape(-1, F)
        Xn = (X2 - MU) / (SD + 1e-6)
        return Xn.reshape(1, T, F).astype("float32")

    def _predict_window(self, Xw: np.ndarray) -> float:
        # Keras
        X = Xw[np.newaxis, ...]
        X = self._norm_apply(X)
        p_keras = float(KERAS.predict(X, verbose=0).ravel()[0])
        if not np.isfinite(p_keras):
            p_keras = 0.0
        p_keras = float(np.clip(p_keras, 0.0, 1.0))

        # LGBM (si existe)
        if LGBM is None or FUSION_W <= 0.0:
            return p_keras

        x3 = Xw.reshape(Xw.shape[0], 17, 3)
        xy = x3[..., :2]
        dx = np.diff(xy, axis=0, prepend=xy[0:1])
        v  = np.linalg.norm(dx, axis=-1)

        def stats(a):
            return np.concatenate([a.mean(0).ravel(),
                                   a.std(0).ravel(),
                                   a.min(0).ravel(),
                                   a.max(0).ravel()], axis=0)

        feat = np.concatenate([stats(xy[...,0]), stats(xy[...,1]), stats(v)], axis=0).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0)
        try:
            p_lgbm = float(LGBM.predict_proba(feat.reshape(1, -1))[:, 1][0])
            if not np.isfinite(p_lgbm): p_lgbm = 0.0
            p_lgbm = float(np.clip(p_lgbm, 0.0, 1.0))
        except Exception as e:
            print("[WARN] LGBM predict err:", e)
            p_lgbm = p_keras

        return (1.0 - FUSION_W) * p_keras + FUSION_W * p_lgbm

    async def run(self):
        import cv2, traceback
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

                frame_full = frame
                p_win = 0.0
                p_vid = 0.0

                try:
                    res = POSE.predict(
                        frame_full, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE,
                        verbose=False, half=False
                    )[0]

                    kps_f = None
                    if (res.keypoints is not None and
                        getattr(res.keypoints, "xy", None) is not None and
                        res.keypoints.xy.shape[0] > 0):
                        xy = res.keypoints.xy.detach().cpu().numpy()     # (P,17,2)
                        c  = (getattr(res.keypoints, "confidence", None) or
                              getattr(res.keypoints, "conf", None))
                        if c is not None:
                            c = c.detach().cpu().numpy()                 # (P,17)
                        else:
                            c = np.ones(xy.shape[:2], dtype=np.float32)

                        # saneo NaNs
                        if not np.isfinite(xy).all(): xy = np.nan_to_num(xy, nan=0.0)
                        if not np.isfinite(c).all():  c  = np.nan_to_num(c,  nan=0.0)

                        order = np.argsort(-c.mean(axis=1))
                        P = int(min(len(order), TOPK))
                        if P > 0:
                            xy = xy[order[:P]]
                            c  = c[order[:P]]
                            kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)  # (P,17,3)

                    feat51 = pool_frame_to_51(kps_f, self.W, self.H)  # (51,)
                    vis = frame_visible(kps_f, CONF_MIN)

                    self.win_feats.append(feat51)
                    self.win_vis.append(1.0 if vis else 0.0)

                    if len(self.win_feats) == SEQ_LEN:
                        vis_frac = float(np.mean(self.win_vis)) if len(self.win_vis) else 0.0
                        if vis_frac >= MIN_VIS_FRAC:
                            Xw = np.stack(self.win_feats, axis=0)  # (T,51)
                            Xw = np.nan_to_num(Xw, nan=0.0)
                            try:
                                p_win = float(self._predict_window(Xw))
                                if not np.isfinite(p_win): p_win = 0.0
                                p_win = float(np.clip(p_win, 0.0, 1.0))
                                self.video_scores.append(p_win)
                                p_vid = pool_scores(list(self.video_scores), pool=POOL_METHOD, topk_frac=TOPK_FRAC)
                            except Exception as e:
                                print("[WARN] predict_window err:", e)
                                traceback.print_exc()
                                p_win = 0.0; p_vid = 0.0

                            # histéresis
                            if not self.on_state and p_vid >= THR_ON:
                                self.on_state = True
                                await self.broadcast({"type": "alert", "cam_id": self.cam_id,
                                                      "prob": float(p_vid), "ts": time.time()})
                            elif self.on_state and p_vid <= THR_OFF:
                                self.on_state = False

                except Exception as e:
                    print("[WARN] pose/pipeline err:", e)
                    import traceback as tb; tb.print_exc()
                    p_win = 0.0; p_vid = 0.0

                # ===== PREVIEW (SÓLO ENVÍO) =====
                try:
                    show = res.plot() if (DRAW_OVERLAY and 'res' in locals() and res is not None) else frame_full
                except Exception:
                    show = frame_full

                # resize sólo para el preview
                if FRAME_WIDTH and show.shape[1] > FRAME_WIDTH:
                    new_h = int(show.shape[0] * (FRAME_WIDTH / show.shape[1]))
                    show_small = await asyncio.to_thread(
                        cv2.resize, show, (FRAME_WIDTH, new_h), cv2.INTER_AREA
                    )
                else:
                    show_small = show

                # JPEG fuera del loop principal
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ok_enc, buf = await asyncio.to_thread(cv2.imencode, ".jpg", show_small, encode_params)
                if not ok_enc:
                    await asyncio.sleep(0.005)
                    continue

                now = time.time()
                send_image = (now - self._last_send_ts) >= (1.0 / max(1.0, SEND_FPS))
                payload = {
                    "type": "frame", "cam_id": self.cam_id,
                    "p_win": float(p_win), "p_vid": float(p_vid),
                    "on": bool(self.on_state), "ts": now,
                }
                if send_image:
                    self._last_send_ts = now
                    payload["jpg_b64"] = base64.b64encode(buf.tobytes()).decode()

                await self.broadcast(payload)

                # telemetría local
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(1e-6, time.time() - t0)); t0 = time.time()
                await asyncio.sleep(0.005)
        finally:
            if cap is not None:
                cap.release()

    async def broadcast(self, msg: Dict[str, Any]):
        # drop rápido de clientes muertos
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                try: await ws.close()
                except Exception: pass
                self.clients.discard(ws)

WORKERS: Dict[str, CameraWorker] = {}

# =========================
# WS (sin JWT)
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
            _ = await websocket.receive_text()  # reservado
    except WebSocketDisconnect:
        worker.clients.discard(websocket)
