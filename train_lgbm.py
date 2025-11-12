# train_lgbm_from_npz.py
import os, json, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import lightgbm as lgb

NPZ_DIRS = [
    r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\out_npz",          # CHAD
    r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\out_npz_fall",     # FALL
    r"C:\Users\Usuario\Documents\GitHub\tesis-deteccionar-sistema\out_npz_video"     # video-level
]
SEQ_LEN      = 32
STRIDE       = 12
MIN_VIS_FRAC = 0.30
CONF_MIN     = 0.10
TRIM_BORDERS = 0
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

OUT_DIR = Path("./models_mix")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / "lgbm_model.pkl"

# === Helpers (idénticos a tu pipeline) ===
def load_meta(z):
    m = z["meta"]
    try: return json.loads(m.item() if hasattr(m, "item") else m)
    except: return {}

def frame_visible(kps_f, conf_min=CONF_MIN):
    conf = np.nan_to_num(kps_f[...,2], nan=0.0)
    return bool((conf >= conf_min).any())

def pool_frame_to_kp(kps_f, W, H):
    J = kps_f.shape[1]  # número de keypoints (17 o 26)
    out = np.zeros((J, 3), np.float32)
    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
    for j in range(J):
        idx = np.argmax(conf_j[:, j]) if conf_j.shape[0] else None
        if idx is not None and conf_j[idx, j] > 0:
            x, y, c = kps_f[idx, j, :]
            if np.isfinite(x) and np.isfinite(y):
                out[j, 0] = np.clip(x / max(W, 1), 0, 1)
                out[j, 1] = np.clip(y / max(H, 1), 0, 1)
                out[j, 2] = float(np.clip(c, 0, 1))
    return out.reshape(-1)  # (J*3,)

def video_to_windows(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    if not {"kps","labels_aligned","meta"}.issubset(z.files):
        return np.zeros((0, SEQ_LEN, 51), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name
    kps = z["kps"]                               # (F,K,17,3)
    y   = z["labels_aligned"].astype(np.int64)   # (F,)
    meta = load_meta(z)
    W, H = int(meta.get("width",1920)), int(meta.get("height",1080))
    F = kps.shape[0]
    if F != len(y) or F < SEQ_LEN:
        return np.zeros((0, SEQ_LEN, 51), np.float32), np.zeros((0,), np.int64), npz_path.name, npz_path.parent.name

    # recorte bordes en clips duros
    if meta.get("video_level_label",0)==1 and TRIM_BORDERS>0 and F>2*TRIM_BORDERS:
        y = y.copy(); y[:TRIM_BORDERS]=0; y[-TRIM_BORDERS:]=0

    J = kps.shape[2]  # número de joints (17 o 26)
    feats = np.zeros((F, J * 3), np.float32)
    vis   = np.zeros((F,), np.float32)
    for t in range(F):
        feats[t] = pool_frame_to_kp(kps[t], W, H)
        vis[t]   = 1.0 if frame_visible(kps[t]) else 0.0

    Xw, Yw = [], []
    for s in range(0, F-SEQ_LEN+1, STRIDE):
        e = s + SEQ_LEN
        if vis[s:e].mean() < MIN_VIS_FRAC: 
            continue
        Xw.append(feats[s:e])         # (T,51) CRUDO (¡así entrenaremos LGBM!)
        Yw.append(int(y[e-1]))        # etiqueta del último frame
    Xw = np.stack(Xw,0) if Xw else np.zeros((0,SEQ_LEN,51), np.float32)
    Yw = np.array(Yw, np.int64) if Yw else np.zeros((0,), np.int64)
    return Xw, Yw, npz_path.name, npz_path.parent.name

PAIR_DISTS = [(5,6),(9,10),(0,9),(0,10),(11,12),(15,16)]
def _stats(a):  
    return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0)], axis=0)

def featurize_TK(X_win: np.ndarray) -> np.ndarray:
    T = X_win.shape[0]
    J = X_win.shape[1] // 3
    xyz = X_win.reshape(T, J, 3)
    x, y, c = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dx = np.diff(x, axis=0, prepend=x[0:1])
    dy = np.diff(y, axis=0, prepend=y[0:1])
    v = np.sqrt(dx * dx + dy * dy)
    Fx, Fy, Fv, Fc = _stats(x), _stats(y), _stats(v), _stats(c)

    # distancias pares entre puntos clave (ajusta según tu dataset)
    pairs = [(5, 6), (11, 12), (23, 24)] if J >= 25 else [(5, 6), (9, 10), (11, 12)]
    D = []
    for (i, j) in pairs:
        dij = np.sqrt((x[:, i] - x[:, j]) ** 2 + (y[:, i] - y[:, j]) ** 2)
        D += [dij.mean(), dij.std()]
    D = np.array(D, np.float32)

    feat = np.concatenate([Fx, Fy, Fv, Fc, D], axis=0).astype(np.float32)
    return feat.reshape(1, -1)


def stratified_split_by_domain(vid_ids, domains, val_ratio, test_ratio, seed=SEED):
    rng = np.random.RandomState(seed)
    vid_ids = np.array(vid_ids); domains = np.array(domains)
    tr, va, te = set(), set(), set()
    for d in np.unique(domains):
        mask = (domains==d)
        vids = np.unique(vid_ids[mask]).copy()
        rng.shuffle(vids)
        n=len(vids); n_te=int(round(n*test_ratio)); n_va=int(round(n*val_ratio))
        te.update(vids[:n_te]); va.update(vids[n_te:n_te+n_va]); tr.update(vids[n_te+n_va:])
    return tr, va, te

# === Carga de ventanas
paths = []
for d in NPZ_DIRS: paths += sorted(Path(d).glob("*.npz"))
assert paths, "No hay .npz"

X_list, y_list, vids, doms = [], [], [], []
for p in paths:
    Xw, Yw, vid, dom = video_to_windows(p)
    if len(Xw):
        X_list.append(Xw); y_list.append(Yw)
        vids += [vid]*len(Yw); doms += [dom]*len(Yw)

X = np.concatenate(X_list,0).astype(np.float32)  # (N,T,51) CRUDO
y = np.concatenate(y_list,0).astype(np.int64)
vids = np.array(vids); doms = np.array(doms)
print(f"Ventanas: {len(y)} | Pos={y.sum()} ({y.mean():.3f})")

# === Split estratificado por dominio (y por video)
train_ids, val_ids, test_ids = stratified_split_by_domain(vids, doms, VAL_RATIO, TEST_RATIO, seed=SEED)
m_tr = np.isin(vids, list(train_ids)); m_va = np.isin(vids, list(val_ids)); m_te = np.isin(vids, list(test_ids))
Xtr, ytr = X[m_tr], y[m_tr]; Xva, yva = X[m_va], y[m_va]; Xte, yte = X[m_te], y[m_te]
print(f"Split: train={len(ytr)} val={len(yva)} test={len(yte)}")

# === Featurización tabular
def batch_feats(Xb):
    return np.concatenate([featurize_TK(w) for w in Xb], axis=0) if len(Xb) else np.zeros((0, 1), np.float32)
Ftr = batch_feats(Xtr); Fva = batch_feats(Xva); Fte = batch_feats(Xte)
print("Shapes feats:", Ftr.shape, Fva.shape, Fte.shape)

# === Pesos de clase (balanced) y entrenamiento
classes = np.array([0,1])
cw = compute_class_weight("balanced", classes=classes, y=ytr)
scale_pos_weight = cw[1]/cw[0]  # proporción (útil en LGBM)

params = dict(
    objective="binary",
    boosting_type="gbdt",
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=1200,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=SEED,
    n_jobs=-1,
    metric="None",
    scale_pos_weight=scale_pos_weight
)

clf = lgb.LGBMClassifier(**params)
clf.fit(
    Ftr, ytr,
    eval_set=[(Fva, yva)],
    eval_metric="auc",
)

# === Validación + Test
for split_name, Fy, yy in [("VAL", Fva, yva), ("TEST", Fte, yte)]:
    proba = clf.predict_proba(Fy)[:,1]
    print(f"\n== {split_name} ==")
    print("ROC-AUC :", roc_auc_score(yy, proba))
    print("PR-AUC  :", average_precision_score(yy, proba))
    thr = 0.5
    pred = (proba >= thr).astype(int)
    print(classification_report(yy, pred, digits=4))

# === Guardar modelo
joblib.dump(clf, OUT_PKL)
print(f"\nGuardado: {OUT_PKL}")
