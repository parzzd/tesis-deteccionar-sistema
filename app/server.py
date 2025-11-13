# uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
import os, json, time, base64, asyncio
from typing import Dict, Any, Optional, Set
from pathlib import Path
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# ============================
# LOGIN SIMPLE
# ============================
SIC_EMAIL = os.environ.get("SIC_EMAIL", "").strip()
SIC_PASSWORD = os.environ.get("SIC_PASSWORD", "").strip()


# ============================
# RUTAS Y PATHS
# ============================
ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
BASE_DIR = ROOT_DIR.parent

KERAS_MODEL = BASE_DIR / "models_mix" / "mix_cnn_lstm_T32_F51.keras"
NORM_STATS = BASE_DIR / "models_mix" / "mix_cnn_lstm_T32_F51_norm_stats.npz"
THRESH_JSON = BASE_DIR / "models_mix" / "mix_cnn_lstm_T32_F51_threshold.json"
LGBM_PKL = BASE_DIR / "models_mix" / "lgbm_model_F51.pkl"

POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolo11m-pose.pt")
IMGSZ = int(os.environ.get("POSE_IMGSZ", "640"))
CONF_POSE = float(os.environ.get("POSE_CONF", "0.25"))
IOU_POSE = float(os.environ.get("POSE_IOU", "0.50"))
TOPK = int(os.environ.get("POSE_TOPK", "4"))

SEQ_LEN = 32
CONF_MIN = 0.10
MIN_VIS_FRAC = 0.30

FUSION_W = 0.50
POOL_METHOD = "topk"
TOPK_FRAC = 0.20
HYST_GAP = 0.10

SEND_FPS = float(os.environ.get("SEND_FPS", "10"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "720"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "65"))
DRAW_OVERLAY = os.environ.get("DRAW_OVERLAY", "1") != "0"
VIDEO_MAX_SCORES = int(os.environ.get("VIDEO_MAX_SCORES", "900"))


# ============================
# UTILS
# ============================
def pool_frame_to_51(kps_f: np.ndarray, W: int, H: int) -> np.ndarray:
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)

    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
    K = kps_f.shape[0]

    for j in range(17):
        if K == 0:
            break
        idx = int(np.argmax(conf_j[:, j]))
        c = conf_j[idx, j]
        if c > 0:
            x, y, _ = kps_f[idx, j, :]
            if np.isfinite(x) and np.isfinite(y):
                out[j, 0] = np.clip(x / max(W, 1), 0.0, 1.0)
                out[j, 1] = np.clip(y / max(H, 1), 0.0, 1.0)
                out[j, 2] = np.clip(c, 0.0, 1.0)

    return out.reshape(-1)


def frame_visible(kps_f: np.ndarray, conf_min=CONF_MIN) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False

    conf = np.nan_to_num(kps_f[..., 2], nan=0.0)
    return bool((conf >= conf_min).any())


def pool_scores(scores, pool="topk", topk_frac=0.20):
    if not scores:
        return 0.0
    arr = np.asarray(scores, np.float32)
    if pool == "max":
        return float(arr.max())
    if pool == "mean":
        return float(arr.mean())

    k = max(1, int(len(arr) * topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())


# ============================
# CARGA MODELOS
# ============================
def load_artifacts():
    keras_model = load_model(str(KERAS_MODEL), compile=False)

    stats = np.load(NORM_STATS)
    mu = stats["mean"].astype("float32")
    sd = stats["std"].astype("float32")

    thr = float(json.loads(Path(THRESH_JSON).read_text()).get("best_threshold", 0.5))

    lgbm = None
    if LGBM_PKL.exists():
        try:
            lgbm = joblib.load(LGBM_PKL)
        except:
            pass

    pose = YOLO(POSE_WEIGHTS)

    return keras_model, mu, sd, thr, lgbm, pose


KERAS, MU, SD, THR_ON, LGBM, POSE = load_artifacts()
THR_OFF = max(0.0, THR_ON - HYST_GAP)

print("[BOOT] Modelos cargados.")


# ============================
# FASTAPI
# ============================
app = FastAPI(title="VigilIA – MVP WebCam + RTSP")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================
# LOGIN
# ============================
@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.post("/login-form")
async def login_form(request: Request):
    form = await request.form()
    email = form.get("email", "").strip()
    password = form.get("password", "").strip()

    if email == SIC_EMAIL and password == SIC_PASSWORD:
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login-fail", status_code=302)


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/login-fail")
def login_fail():
    return FileResponse(str(STATIC_DIR / "login_fail.html"))


@app.get("/health")
def health():
    return {"ok": True}


# ============================
# CÁMARAS RTSP / HTTP / archivo
# ============================
class CameraConfig(BaseModel):
    cam_id: str
    src: str


CAMERAS: Dict[str, CameraConfig] = {}


@app.get("/cameras")
def list_cameras():
    return list(CAMERAS.values())


@app.post("/cameras")
def add_camera(cfg: CameraConfig):
    CAMERAS[cfg.cam_id] = cfg
    return {"ok": True}


@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str):
    CAMERAS.pop(cam_id, None)
    return {"ok": True}


# ============================
# WORKER
# ============================
class CameraWorker:

    def __init__(self, cam_id, src):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        self.win_feats = deque(maxlen=SEQ_LEN)
        self.win_vis = deque(maxlen=SEQ_LEN)
        self.video_scores = deque(maxlen=VIDEO_MAX_SCORES)

        self.on_state = False
        self.W, self.H = None, None
        self._last_send = 0

    # ------------------------------------------
    # NORMALIZADOR
    # ------------------------------------------
    def _norm_apply(self, X):
        T, F = X.shape[1], X.shape[2]
        flat = X.reshape(-1, F)
        norm = (flat - MU) / (SD + 1e-6)
        return norm.reshape(1, T, F).astype("float32")

    # ------------------------------------------
    # PREDICCIÓN
    # ------------------------------------------
    def _predict_window(self, Xw):
        X = self._norm_apply(Xw)
        p_keras = float(KERAS.predict(X, verbose=0).ravel()[0])
        p_keras = float(np.clip(p_keras, 0, 1))

        if LGBM is None or FUSION_W <= 0:
            return p_keras

        x3 = Xw.reshape(Xw.shape[0], 17, 3)
        xy = x3[..., :2]
        dx = np.diff(xy, axis=0, prepend=xy[:1])
        v = np.linalg.norm(dx, axis=-1)

        def stats(a):
            return np.concatenate([
                a.mean(0).ravel(),
                a.std(0).ravel(),
                a.min(0).ravel(),
                a.max(0).ravel()
            ])

        feat = np.concatenate([stats(xy[..., 0]), stats(xy[..., 1]), stats(v)])
        feat = np.nan_to_num(feat).astype(np.float32)

        try:
            p_lgbm = float(LGBM.predict_proba(feat.reshape(1, -1))[:, 1][0])
        except:
            p_lgbm = p_keras

        return (1 - FUSION_W) * p_keras + FUSION_W * p_lgbm

    # ------------------------------------------
    # PROCESAR UN FRAME (para webcam)
    # ------------------------------------------
    async def process_frame(self, frame):
        await self._process_frame_logic(frame)

    # ------------------------------------------
    # PROCESAR FRAME (comparten RTSP y webcam)
    # ------------------------------------------
    async def _process_frame_logic(self, frame):
        import cv2

        frame_full = frame
        p_win = 0.0
        p_vid = 0.0

        try:
            res = POSE.predict(frame_full, imgsz=IMGSZ, conf=CONF_POSE,
                               iou=IOU_POSE, verbose=False, half=False)[0]

            kps_f = None
            if res.keypoints is not None and res.keypoints.xy is not None:
                xy = res.keypoints.xy.detach().cpu().numpy()
                c = (getattr(res.keypoints, "confidence", None)
                     or getattr(res.keypoints, "conf", None))

                if c is not None:
                    c = c.detach().cpu().numpy()
                else:
                    c = np.ones(xy.shape[:2], np.float32)

                if not np.isfinite(xy).all():
                    xy = np.nan_to_num(xy)
                if not np.isfinite(c).all():
                    c = np.nan_to_num(c)

                order = np.argsort(-c.mean(axis=1))
                P = min(len(order), TOPK)

                if P > 0:
                    xy = xy[order[:P]]
                    c = c[order[:P]]
                    kps_f = np.concatenate([xy, c[..., None]], axis=-1)

            feat51 = pool_frame_to_51(kps_f, self.W, self.H)
            vis = frame_visible(kps_f)

            self.win_feats.append(feat51)
            self.win_vis.append(1 if vis else 0)

            if len(self.win_feats) == SEQ_LEN:
                vis_frac = float(np.mean(self.win_vis))
                if vis_frac >= MIN_VIS_FRAC:
                    Xw = np.stack(self.win_feats)
                    Xw = np.nan_to_num(Xw)

                    p_win = self._predict_window(Xw)
                    self.video_scores.append(p_win)
                    p_vid = pool_scores(self.video_scores, POOL_METHOD, TOPK_FRAC)

                    if not self.on_state and p_vid >= THR_ON:
                        self.on_state = True
                        await self.broadcast({"type": "alert",
                                              "cam_id": self.cam_id,
                                              "prob": float(p_vid),
                                              "ts": time.time()})
                    elif self.on_state and p_vid <= THR_OFF:
                        self.on_state = False

        except:
            pass

        # enviar preview
        try:
            show = res.plot() if DRAW_OVERLAY else frame_full
        except:
            show = frame_full

        # resize
        if FRAME_WIDTH and show.shape[1] > FRAME_WIDTH:
            new_h = int(show.shape[0] * (FRAME_WIDTH / show.shape[1]))
            show = cv2.resize(show, (FRAME_WIDTH, new_h))

        ok, buf = cv2.imencode(".jpg", show, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return

        payload = {
            "type": "frame",
            "cam_id": self.cam_id,
            "p_win": float(p_win),
            "p_vid": float(p_vid),
            "on": bool(self.on_state),
            "ts": time.time(),
            "jpg_b64": base64.b64encode(buf.tobytes()).decode()
        }

        await self.broadcast(payload)

    # ------------------------------------------
    # BROADCAST
    # ------------------------------------------
    async def broadcast(self, msg):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except:
                try:
                    await ws.close()
                except:
                    pass
                self.clients.discard(ws)

    # ------------------------------------------
    # LOOP PARA RTSP
    # ------------------------------------------
    async def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        import cv2

        if self.src.strip().isdigit():
            cap = cv2.VideoCapture(int(self.src.strip()))
        else:
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            await self.broadcast({"type": "error", "msg": "no-open"})
            return

        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                await self._process_frame_logic(frame)
                await asyncio.sleep(0.005)
        finally:
            cap.release()


# ============================
# WORKERS
# ============================
WORKERS: Dict[str, CameraWorker] = {}


# ============================
# WEBCAM DEL NAVEGADOR
# ============================
@app.websocket("/ws/webcam")
async def ws_webcam(websocket: WebSocket):
    await websocket.accept()

    cam_id = "webcam"

    if cam_id not in WORKERS:
        WORKERS[cam_id] = CameraWorker(cam_id, src="webcam")
        WORKERS[cam_id].W = 640
        WORKERS[cam_id].H = 480

    worker = WORKERS[cam_id]
    worker.clients.add(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue

            import cv2
            jpg = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(jpg, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            await worker.process_frame(frame)

    except WebSocketDisconnect:
        worker.clients.discard(websocket)


# ============================
# WS RTSP / HTTP
# ============================
@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(websocket: WebSocket, cam_id: str):
    await websocket.accept()

    cfg = CAMERAS.get(cam_id)
    if cfg is None:
        await websocket.send_json({"type": "error", "msg": "cam-not-found"})
        await websocket.close()
        return

    if cam_id not in WORKERS:
        WORKERS[cam_id] = CameraWorker(cam_id, cfg.src)
        await WORKERS[cam_id].start()

    worker = WORKERS[cam_id]
    worker.clients.add(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(websocket)
