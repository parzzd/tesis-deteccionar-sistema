# infer_video_simple.py
# Ejecuta inferencia (Keras + opcional LGBM) sobre un video y guarda MP4 anotado + CSV.

import os, json, csv, time, collections, warnings
from pathlib import Path
from typing import List
import numpy as np
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# =========================
# CONFIG (edita aquí)
# =========================
MODELS_DIR   = Path("./models_mix")
VIDEO_IN     = Path(r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\V_27.mp4")
VIDEO_OUT    = Path(r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\V_222.mp4")
CSV_OUT      = VIDEO_OUT.with_suffix(".csv")

POSE_WEIGHTS = "yolo11m-pose.pt"
IMGSZ        = 920
CONF_POSE    = 0.25
IOU_POSE     = 0.50
TOPK_PERSONS = 4

# Ventana/modelo (debe coincidir con tu entrenamiento)
SEQ_LEN      = 32
CONF_MIN     = 0.10
MIN_VIS_FRAC = 0.30
HYST_GAP     = 0.10    # thr_off = thr_on - gap
STRIDE       = 1       # procesa 1 frame cada STRIDE (2 = saltea 1)

# Fusión con LGBM (si existe models_mix/lgbm_model.pkl)
FUSION_W     = 0.50    # 0→solo Keras, 1→solo LGBM

# Pooling de scores a nivel video
POOL_METHOD  = "topk"  # "max" | "mean" | "topk"
TOPK_FRAC    = 0.20

# =========================
# Rutas de artefactos
# =========================
KERAS_FILE = MODELS_DIR / "mix_cnn_lstm_T32_F51.keras"
STATS_FILE = MODELS_DIR / "mix_cnn_lstm_T32_F51_norm_stats.npz"
THR_FILE   = MODELS_DIR / "mix_cnn_lstm_T32_F51_threshold.json"
LGBM_FILE  = MODELS_DIR / "lgbm_model.pkl"

# Warnings ruidosos
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =========================
# Utilidades
# =========================
def pool_frame_to_51(kps_f: np.ndarray, W: int, H: int) -> np.ndarray:
    """kps_f: (P,17,3) -> (51,) normalizado por W,H; pooling por joint."""
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)
    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
    P = kps_f.shape[0]
    for j in range(17):
        if P == 0: break
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

def build_tabular_features(Xw: np.ndarray) -> np.ndarray:
    """
    Genera features tabulares desde Xw (T,51) reinterpretando como (T,17,3):
    - stats de x, y y velocidad (mean,std,min,max) → vector (actualmente 204 dims).
    OJO: debe coincidir con el featurizador usado en el entrenamiento del LGBM.
    """
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
    return feat  # p.ej. 204 dims con este diseño

def lgbm_expected_features(lgbm) -> int | None:
    """Intenta obtener cuántas features espera el LGBM."""
    exp = getattr(lgbm, "n_features_in_", None)
    if exp is not None:
        return int(exp)
    try:
        if hasattr(lgbm, "booster_"):
            names = lgbm.booster_.feature_name()
            if names:
                return len(names)
    except Exception:
        pass
    return None

def load_artifacts(models_dir: Path, pose_w: str):
    if not KERAS_FILE.exists(): raise FileNotFoundError(KERAS_FILE)
    if not STATS_FILE.exists(): raise FileNotFoundError(STATS_FILE)
    keras_model = load_model(str(KERAS_FILE), compile=False)
    stats = np.load(STATS_FILE)
    MU = stats["mean"].astype("float32")
    SD = stats["std"].astype("float32")
    if THR_FILE.exists():
        THR_ON = float(json.loads(THR_FILE.read_text(encoding="utf-8")).get("best_threshold", 0.5))
    else:
        THR_ON = 0.5
    THR_OFF = max(0.0, THR_ON - HYST_GAP)

    lgbm = None
    if LGBM_FILE.exists():
        try:
            lgbm = joblib.load(LGBM_FILE)
            print(f"[BOOT] LGBM ON → {LGBM_FILE}")
            exp = lgbm_expected_features(lgbm)
            if exp is not None:
                print(f"[LGBM] Espera {exp} features de entrada")
        except Exception as e:
            print(f"[BOOT] LGBM fallo al cargar ({e}), continuo sin LGBM")

    pose = YOLO(pose_w)
    print(f"[BOOT] Keras={KERAS_FILE} | THR_ON={THR_ON:.2f} THR_OFF={THR_OFF:.2f} | T={SEQ_LEN} | LGBM={'ON' if lgbm is not None else 'OFF'}")
    return keras_model, MU, SD, THR_ON, THR_OFF, lgbm, pose

def norm_apply(X, MU, SD):
    T, F = X.shape[1], X.shape[2]
    X2 = X.reshape(-1, F)
    Xn = (X2 - MU) / (SD + 1e-6)
    return Xn.reshape(1, T, F).astype("float32")

def predict_window(Xw, keras_model, MU, SD, lgbm=None, fusion_w=0.5):
    # --- Keras ---
    X = Xw[np.newaxis, ...]
    X = norm_apply(X, MU, SD)
    p_keras = float(keras_model.predict(X, verbose=0).ravel()[0])

    # --- LGBM (opcional) ---
    if lgbm is None or fusion_w <= 0.0:
        return p_keras

    feat = build_tabular_features(Xw)              # tu featurizador actual
    expected = lgbm_expected_features(lgbm)        # lo que espera el modelo
    if expected is not None and feat.shape[0] != expected:
        # Fallback limpio (no romper): usa solo Keras
        print(f"[WARN] LGBM mismatch: generas {feat.shape[0]} feats y el modelo espera {expected}. "
              f"Usaré solo Keras en esta corrida.")
        return p_keras

    try:
        p_lgbm = float(lgbm.predict_proba(feat.reshape(1, -1))[:, 1][0])
    except Exception as e:
        print(f"[WARN] LGBM predict_proba falló ({e}). Uso solo Keras.")
        return p_keras

    return (1.0 - fusion_w) * p_keras + fusion_w * p_lgbm

# =========================
# Inferencia sobre video
# =========================
def run_video():
    KERAS, MU, SD, THR_ON, THR_OFF, LGBM, POSE = load_artifacts(MODELS_DIR, POSE_WEIGHTS)

    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (W, H), True)

    win_feats = collections.deque(maxlen=SEQ_LEN)  # (51,)
    win_vis   = collections.deque(maxlen=SEQ_LEN)  # bool
    video_scores: List[float] = []
    on_state = False

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame_idx", "time_sec", "p_win", "p_vid", "on"])

        fidx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if STRIDE > 1 and (fidx % STRIDE != 0):
                # mantén sincronía de tiempo en el MP4 de salida
                vw.write(frame)
                continue

            # --- Pose ---
            res = POSE.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE, verbose=False, half=False)[0]

            kps_f = None
            if res.keypoints is not None and res.keypoints.xy is not None and res.keypoints.xy.shape[0] > 0:
                xy = res.keypoints.xy.detach().cpu().numpy()  # (P,17,2)
                c  = getattr(res.keypoints, "confidence", None) or getattr(res.keypoints, "conf", None)
                if c is not None:
                    c = c.detach().cpu().numpy()              # (P,17)
                else:
                    c = np.ones(xy.shape[:2], dtype=np.float32)
                order = np.argsort(-c.mean(axis=1))
                P = min(len(order), TOPK_PERSONS)
                xy = xy[order[:P]]
                c  = c[order[:P]]
                kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)  # (P,17,3)

            # --- Ventana ---
            feat51 = pool_frame_to_51(kps_f, W, H)
            vis = frame_visible(kps_f, CONF_MIN)
            win_feats.append(feat51)
            win_vis.append(1.0 if vis else 0.0)

            p_win = 0.0
            p_vid = 0.0
            if len(win_feats) == SEQ_LEN:
                vis_frac = np.mean(win_vis)
                if vis_frac >= MIN_VIS_FRAC:
                    Xw = np.stack(win_feats, axis=0)  # (T,51)
                    p_win = predict_window(Xw, KERAS, MU, SD, lgbm=LGBM, fusion_w=FUSION_W)
                    video_scores.append(p_win)
                    p_vid = pool_scores(video_scores, pool=POOL_METHOD, topk_frac=TOPK_FRAC)

                    if (not on_state) and p_vid >= THR_ON:
                        on_state = True
                    elif on_state and p_vid <= THR_OFF:
                        on_state = False

            # --- Render ---
            try:
                show = res.plot() if res is not None else frame
            except Exception:
                show = frame

            cv2.putText(show, f"p_win={p_win:.2f}  p_vid={p_vid:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 2)
            if on_state:
                cv2.putText(show, "ALERTA", (20, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)

            vw.write(show)

            tsec = fidx / max(fps, 1e-6)
            csv_w.writerow([fidx, f"{tsec:.3f}", f"{p_win:.6f}", f"{p_vid:.6f}", int(on_state)])

    vw.release()
    cap.release()
    print(f"[DONE] Video: {VIDEO_OUT}")
    print(f"[DONE] CSV  : {CSV_OUT}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    run_video()
