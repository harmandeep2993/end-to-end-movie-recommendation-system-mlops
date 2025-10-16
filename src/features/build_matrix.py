import os, json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

INTERIM = "data/interim"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

# pick the largest ratings_sample_*.csv in data/interim
samples = [f for f in os.listdir(INTERIM) if f.startswith("ratings_sample_")]
assert samples, "No sample found. Run make_sample.py first."
SAMPLE = max(samples, key=lambda x: int(x.split("_")[-1].split(".")[0]))
SRC = os.path.join(INTERIM, SAMPLE)

MIN_EVENTS = 5
POS_THRESH = 3.5

print("Load:", SRC)
use = ["userId","movieId","rating","timestamp"]
df = pd.read_csv(SRC, usecols=use)

# implicit positives
df = df[df["rating"] >= POS_THRESH].copy()

# filter users with enough history
vc = df["userId"].value_counts()
keep_users = vc[vc >= MIN_EVENTS].index
df = df[df["userId"].isin(keep_users)].copy()

# index encode
df["uid"] = df["userId"].astype("category").cat.codes
df["iid"] = df["movieId"].astype("category").cat.codes
n_users = int(df["uid"].max() + 1)
n_items = int(df["iid"].max() + 1)
print(f"Users={n_users} Items={n_items} Interactions={len(df)}")

# time-based split per user
df = df.sort_values(["uid","timestamp"])
valid_idx = df.groupby("uid").tail(1).index
valid = df.loc[valid_idx]
train = df.drop(valid_idx)

def to_csr(d):
    return csr_matrix(
        (np.ones(len(d), dtype=np.float32), (d["uid"].to_numpy(), d["iid"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

ui_train = to_csr(train)
ui_valid = to_csr(valid)

# save matrices
save_npz(os.path.join(PROC, "user_item_train.npz"), ui_train)
save_npz(os.path.join(PROC, "user_item_valid.npz"), ui_valid)

# save id maps
maps_dir = os.path.join(PROC, "mappings"); os.makedirs(maps_dir, exist_ok=True)
df[["userId","uid"]].drop_duplicates().sort_values("uid").to_csv(
    os.path.join(maps_dir, "user_map.csv"), index=False
)
df[["movieId","iid"]].drop_duplicates().sort_values("iid").to_csv(
    os.path.join(maps_dir, "item_map.csv"), index=False
)

# metadata
meta = {
    "n_users": n_users,
    "n_items": n_items,
    "n_train": int(ui_train.nnz),
    "n_valid": int(ui_valid.nnz),
    "pos_threshold": POS_THRESH,
    "min_events": MIN_EVENTS,
    "source_file": SRC,
}
json.dump(meta, open(os.path.join(PROC, "meta.json"), "w"))
print("Saved to data/processed")
print(meta)