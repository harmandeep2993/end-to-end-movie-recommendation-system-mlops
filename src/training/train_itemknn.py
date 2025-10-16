# src/training/train_itemknn.py
import os, json, time, math
import numpy as np
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
import joblib
from functools import lru_cache

PROC = "data/processed"
MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)

t0 = time.perf_counter()
UI_TRAIN = load_npz(os.path.join(PROC, "user_item_train.npz")).tocsr()
UI_VALID = load_npz(os.path.join(PROC, "user_item_valid.npz")).tocsr()
print(f"[load] train {UI_TRAIN.shape}, nnz={UI_TRAIN.nnz}")
print(f"[load] valid {UI_VALID.shape}, nnz={UI_VALID.nnz}")

ITEM_MATRIX = UI_TRAIN.T.tocsr()
density_train = UI_TRAIN.nnz / (UI_TRAIN.shape[0] * UI_TRAIN.shape[1])
density_items = ITEM_MATRIX.nnz / (ITEM_MATRIX.shape[0] * ITEM_MATRIX.shape[1])
print(f"[info] item_matrix {ITEM_MATRIX.shape}, density={density_items:.8f}")
print(f"[info] user_item   density={density_train:.8f}")

# ---- fit KNN (unchanged k=50) ----
k_neighbors = 50
print(f"[fit] NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors={k_neighbors})")
t1 = time.perf_counter()
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_neighbors, n_jobs=-1)
knn.fit(ITEM_MATRIX)
print(f"[fit] done in {time.perf_counter()-t1:.2f}s")

joblib.dump(knn, os.path.join(MODEL_DIR, "itemknn.joblib"))
print("[save] models/itemknn.joblib")

# ---- recommend and eval (same logic, now with progress) ----
def _seen(uid: int):
    s, e = UI_TRAIN.indptr[uid], UI_TRAIN.indptr[uid + 1]
    return UI_TRAIN.indices[s:e]

@lru_cache(maxsize=100_000)
def _neighbors(it: int):
    dists, idxs = knn.kneighbors(ITEM_MATRIX[it], n_neighbors=k_neighbors, return_distance=True)
    sims = 1.0 - dists.ravel()
    return idxs.ravel(), sims

def recommend(uid: int, K: int = 10):
    user_items = _seen(uid)
    if user_items.size == 0: return []
    scores = {}
    for it in user_items:
        idxs, sims = _neighbors(int(it))
        for j, s in zip(idxs, sims):
            if j in user_items: continue
            scores[j] = scores.get(j, 0.0) + float(s)
    if not scores: return []
    return [j for j, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]]

# quick Hit@10 with progress
t2 = time.perf_counter()
users = np.where(UI_VALID.getnnz(axis=1) > 0)[0]
hits = 0
report_every = max(1, len(users)//10)
print(f"[eval] users={len(users)}  report_every={report_every}")
for i, u in enumerate(users, 1):
    s, e = UI_VALID.indptr[u], UI_VALID.indptr[u+1]
    target = set(UI_VALID.indices[s:e])
    recs = set(recommend(int(u), 10))
    if target & recs: hits += 1
    if i % report_every == 0 or i == len(users):
        elapsed = time.perf_counter() - t2
        rate = i/elapsed if elapsed > 0 else math.inf
        print(f"[eval] {i}/{len(users)} users | hits={hits} | {rate:.1f} users/s")

hit_at_10 = hits / len(users) if len(users) else 0.0
metrics = {
    "users_eval": int(len(users)),
    "hit@10": hit_at_10,
    "neighbors": k_neighbors,
    "t_fit_sec": round(time.perf_counter()-t1, 2),
    "t_total_sec": round(time.perf_counter()-t0, 2),
}
with open(os.path.join(PROC, "metrics_itemknn.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("[done]", metrics)