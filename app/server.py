
#uvicorn server:app --reload --host 0.0.0.0 --port 8000


import os, io, cv2, jwt, time, base64, asyncio, json, collections
from typing import Dict, Any, Optional, Set
import numpy as np

import joblib
import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# =========================
# Config
# =========================

DATA_DIR     = os.environ.get("SIC_DATA_DIR", r"C:\tesis\pose_sequences")
MODEL_PT     = os.path.join(DATA_DIR, "modelo.pt")
LGBM_CAL_PKL = os.path.join(DATA_DIR, "lgbm_calibrated.pkl")   # bundle calibrado (dict o estimador)
LGBM_PKL     = os.path.join(DATA_DIR, "lgbm_model.pkl")        # fallback si no hay calibrado
POSE_WEIGHTS = os.environ.get("SIC_POSE", "yolov8n-pose.pt")
TRACKER_CFG  = "bytetrack.yaml"

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ     = int(os.environ.get("SIC_IMGSZ", "640"))
CONF_POSE = float(os.environ.get("SIC_CONF", "0.25"))

# Histeresis / pooling / penalizaciones
HYST_GAP             = 0.10          # diferencia on/off (se usa sobre fusion_thr_video)
MIN_CONSEC_ON        = 2             # ventanas consecutivas para activar
MIN_PERSONS_FOR_FULL = 2             # penaliza si hay 1 sola persona
SOLO_PERSON_PENALTY  = 0.85

# JWT (demo)
JWT_SECRET   = os.environ.get("SIC_JWT_SECRET", "devsecret")
JWT_ALG      = "HS256"
SIC_EMAIL    = os.environ.get("SIC_EMAIL", "    ")
SIC_PWD      = os.environ.get("SIC_PASSWORD")       # plano (demo)
SIC_PWD_HASH = os.environ.get("SIC_PASSWORD_HASH")  # bcrypt (prod)

# =========================
# Modelo LSTM (igual que en train)
# =========================

class AttnPool(nn.Module):
    def __init__(self, dim): 
        super().__init__(); self.proj = nn.Linear(dim, 1)
    def forward(self, H):      # (B,T,D)
        a = torch.softmax(self.proj(H).squeeze(-1), dim=1)
        return (H * a.unsqueeze(-1)).sum(1)

class KeypointLSTM(nn.Module):
    def __init__(self, in_dim, hid=192, bidir=True, dropout_head=0.35):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True,
                            bidirectional=bidir, dropout=0.0)
        D = hid*(2 if bidir else 1)
        self.pool = AttnPool(D)
        self.head = nn.Sequential(
            nn.Linear(D, 192), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(192, 2)
        )
        self.temp_scale = nn.Parameter(torch.ones(1))
    def forward(self, x):      # x: (B,T,D)
        H,_ = self.lstm(x)
        h = self.pool(H)
        logits = self.head(h) / self.temp_scale.clamp(min=0.5, max=2.0)
        return logits

# =========================
# Featurizadores / utils — (2,T,17)
# =========================

def norm_kpts_by_bbox(kpts_xy, bb):
    x1,y1,x2,y2 = bb
    w = max(1e-6, x2-x1); h = max(1e-6, y2-y1)
    k = kpts_xy.astype(np.float32).copy()
    k[:,0] = (k[:,0]-x1)/w
    k[:,1] = (k[:,1]-y1)/h
    return np.clip(k, 0.0, 1.0)

def build_lstm_input(seq_2_T_17: np.ndarray, use_delta: bool) -> torch.Tensor:
    T = seq_2_T_17.shape[1]
    x = torch.from_numpy(seq_2_T_17).float().permute(1,0,2).reshape(T, -1)  # (T, 34)
    if use_delta:
        dx = torch.zeros_like(x)
        dx[1:] = x[1:] - x[:-1]
        x = torch.cat([x, dx], dim=1)  # (T, 68)
    return x.unsqueeze(0)  # (1, T, D)

PAIR_DISTS = [(5,6),(9,10),(0,9),(0,10),(11,12),(15,16)]
def featurize_seq_for_lgbm(seq_2_T_17: np.ndarray) -> np.ndarray:
    # (2,T,17) -> vector tabular (igual a train)
    x = seq_2_T_17[0]     # (T,17)
    yk = seq_2_T_17[1]
    dx = np.diff(x, axis=0, prepend=x[0:1])
    dy = np.diff(yk,axis=0, prepend=yk[0:1])
    v  = np.sqrt(dx*dx + dy*dy)

    def stats(a):
        return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0)], axis=0)

    Fx, Fy, Fv = stats(x), stats(yk), stats(v)

    def pair_dist(i,j):
        d = np.sqrt((x[:,i]-x[:,j])**2 + (yk[:,i]-yk[:,j])**2)
        return np.array([d.mean(), d.std()], dtype=np.float32)

    D = np.concatenate([pair_dist(i,j) for (i,j) in PAIR_DISTS], axis=0)  # len=12
    feat = np.concatenate([Fx, Fy, Fv, D], axis=0).astype(np.float32)
    return feat.reshape(1, -1)

def pool_scores(scores, pool="topk", topk_frac=0.2):
    if len(scores)==0: return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if pool=="max":  return float(arr.max())
    if pool=="mean": return float(arr.mean())
    k = max(1, int(len(arr)*topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())

# =========================
# Carga de artefactos
# =========================

def load_artifacts():
    # YOLO pose
    pose = YOLO(POSE_WEIGHTS)

    # LSTM + meta
    ckpt = torch.load(MODEL_PT, map_location="cpu")
    meta = ckpt["meta"]
    in_dim = int(meta["in_dim"])
    model = KeypointLSTM(in_dim=in_dim, hid=meta["hid"], bidir=meta["bidir"],
                         dropout_head=meta["dropout_head"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # fusión/umbrales/pooling
    pool = meta.get("pool","topk")
    topk_frac = meta.get("topk_frac", 0.2)
    thr_video_lstm = float(meta.get("threshold_video", meta.get("threshold", 0.5)))
    fusion_w = float(meta.get("fusion_w", 1.0))                # 1.0 => solo LSTM
    fusion_thr_video = float(meta.get("fusion_thr_video", thr_video_lstm))
    use_delta = bool(meta.get("USE_DELTA", True))
    T = int(meta["T"])

    # LGBM calibrado o normal
    cal = None
    tried = []
    for path in [LGBM_CAL_PKL, LGBM_PKL]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                cal = obj if isinstance(obj, dict) else obj
                print(f"[INFO] LGBM cargado: {path}")
                break
            except Exception as e:
                tried.append((path, str(e)))
    if cal is None and tried:
        print("[WARN] No se pudo cargar LGBM:", tried)

    return pose, model, meta, cal, pool, topk_frac, thr_video_lstm, fusion_w, fusion_thr_video, use_delta, T

pose_model, lstm_model, META, LGBM_CAL, POOL_METH, TOPK_FRAC, THR_LSTM_VIDEO, FUSION_W, FUSION_THR_VIDEO, USE_DELTA, T = load_artifacts()

print(f"[BOOT] Device={DEVICE} | T={T} | pool={POOL_METH} topk={TOPK_FRAC} | thr_lstm={THR_LSTM_VIDEO:.3f} fusion_w={FUSION_W:.2f} fusion_thr={FUSION_THR_VIDEO:.3f}")

# =========================
# Auth (usuario único) con JWT
# =========================

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

app = FastAPI(title="Sicher Server")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    ok = False
    if data.email == SIC_EMAIL:
        if SIC_PWD_HASH:
            import bcrypt
            ok = bcrypt.checkpw(data.password.encode(), SIC_PWD_HASH.encode())
        elif SIC_PWD:
            ok = (data.password == SIC_PWD)
    if not ok:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    payload = {"sub": data.email, "iat": int(time.time()), "exp": int(time.time()) + 12*3600}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return TokenResponse(access_token=token)

# =========================
# Gestión de cámaras
# =========================

class CameraConfig(BaseModel):
    cam_id: str
    src: str  # "0"/"1"/rtsp/http/file path

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
# Worker y WebSocket
# =========================

class CameraWorker:
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # por persona (track id) → deque de (2,17) por frame
        self.per_id_seq: Dict[int, collections.deque] = {}
        self.per_id_scores: Dict[int, list] = {}
        self.per_id_streak: Dict[int, int] = {}
        self.alerts_on: Dict[int, bool] = {}

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

    async def run(self):
        cap = None
        try:
            if self.src.strip().isdigit():
                cap = cv2.VideoCapture(int(self.src.strip()))
            else:
                cap = cv2.VideoCapture(self.src)
                if not cap.isOpened():
                    # intentar FFMPEG
                    cap.release()
                    cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                await self.broadcast({"type":"error","msg":"no-open"})
                return

            fps, t0 = 0.0, time.time()
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    await self.broadcast({"type":"error","msg":"eof"})
                    break

                # Pose + ByteTrack
                res = pose_model.track(frame, imgsz=IMGSZ, conf=CONF_POSE,
                                       tracker="bytetrack.yaml", persist=True,
                                       verbose=False, device=0 if DEVICE=="cuda" else "cpu",
                                       half=(DEVICE=="cuda"))[0]

                p_win_disp, p_vid_disp, alert_disp = 0.0, 0.0, False
                n_persons = 0

                if res.boxes is not None and res.keypoints is not None and res.boxes.cls is not None:
                    cls = res.boxes.cls.int().cpu().numpy()
                    person = (cls == 0)
                    if person.any() and res.boxes.id is not None:
                        xyxy = res.boxes.xyxy.cpu().numpy()[person]
                        ids  = res.boxes.id.int().cpu().numpy()[person]
                        has_k = (res.keypoints.xy is not None)
                        kpts = res.keypoints.xy.cpu().numpy()[person] if has_k else None
                        n_persons = len(ids)

                        for j, tid in enumerate(ids):
                            if kpts is None: continue
                            bb = xyxy[j]
                            k_xy = kpts[j]                # (17,2)
                            nk = norm_kpts_by_bbox(k_xy, bb)  # (17,2) en [0,1]

                            dq = self.per_id_seq.get(tid)
                            if dq is None:
                                dq = collections.deque(maxlen=T); self.per_id_seq[tid] = dq
                                self.per_id_scores[tid] = []; self.per_id_streak[tid] = 0; self.alerts_on[tid] = False
                            dq.append(nk.T)  # (2,17) por frame

                            if len(dq) == T:
                                # (2,T,17)
                                seq = np.stack(list(dq), axis=0).transpose(1,0,2).astype(np.float32)

                                # --- LSTM ---
                                x = build_lstm_input(seq, USE_DELTA).to(DEVICE)  # (1,T,D)
                                with torch.no_grad():
                                    logits = lstm_model(x)
                                    p_lstm = torch.softmax(logits, dim=1)[:,1].item()

                                # --- LGBM (si disponible) ---
                                if LGBM_CAL is not None:
                                    F = featurize_seq_for_lgbm(seq)   # (1,D_tab)
                                    clf = LGBM_CAL["clf"] if isinstance(LGBM_CAL, dict) and "clf" in LGBM_CAL else LGBM_CAL
                                    p_lgbm = float(clf.predict_proba(F)[:,1][0])
                                else:
                                    p_lgbm = p_lstm

                                # --- fusión tardía (ventana) ---
                                p_win = (1.0 - FUSION_W) * p_lstm + (FUSION_W) * p_lgbm
                                if n_persons < MIN_PERSONS_FOR_FULL:
                                    p_win *= SOLO_PERSON_PENALTY

                                self.per_id_scores[tid].append(p_win)
                                p_vid = pool_scores(self.per_id_scores[tid], pool=POOL_METH, topk_frac=TOPK_FRAC)

                                # Histeresis usando threshold de fusión
                                thr_on  = FUSION_THR_VIDEO
                                thr_off = max(0.0, FUSION_THR_VIDEO - HYST_GAP)

                                if not self.alerts_on[tid]:
                                    if p_vid >= thr_on:
                                        self.per_id_streak[tid] += 1
                                        if self.per_id_streak[tid] >= MIN_CONSEC_ON:
                                            self.alerts_on[tid] = True
                                            self.per_id_streak[tid] = 0
                                            await self.broadcast({"type":"alert","cam_id":self.cam_id,"prob":float(p_vid),"ts":time.time()})
                                    else:
                                        self.per_id_streak[tid] = 0
                                else:
                                    if p_vid <= thr_off:
                                        self.alerts_on[tid] = False
                                        self.per_id_streak[tid] = 0

                                # para overlay
                                p_win_disp = max(p_win_disp, float(p_win))
                                p_vid_disp = max(p_vid_disp, float(p_vid))
                                alert_disp = alert_disp or self.alerts_on[tid]

                # plot + HUD
                try:
                    show = res.plot() if res is not None else frame
                except Exception:
                    show = frame

                fps = 0.9*fps + 0.1*(1.0/max(1e-6, time.time()-t0)); t0 = time.time()
                cv2.putText(show, f"fps={fps:.1f}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)
                cv2.putText(show, f"p_win={p_win_disp:.2f}  p_vid={p_vid_disp:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)
                if alert_disp:
                    cv2.putText(show, "ALERTA", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 3)

                # enviar frame a clientes
                _, buf = cv2.imencode('.jpg', show)
                b64 = base64.b64encode(buf.tobytes()).decode()
                payload = {
                    "type":"frame", "cam_id":self.cam_id, "p_win":p_win_disp, "p_vid":p_vid_disp,
                    "on":bool(alert_disp), "ts":time.time(), "jpg_b64":b64
                }
                await self.broadcast(payload)

                # rate limit mínimo (ajusta si quieres más fps)
                await asyncio.sleep(0.01)
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

# --- Auth para WS ---
async def verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido")

@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(websocket: WebSocket, cam_id: str, token: Optional[str] = None):
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION); return
    try:
        _user = await verify_token(token)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION); return

    await websocket.accept()
    cfg = CAMERAS.get(cam_id)
    if cfg is None:
        await websocket.send_json({"type":"error","msg":"cam-not-found"})
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
            # opcional: recibir comandos (p.ej. cambiar thresholds)
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(websocket)
