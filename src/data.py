"""Descarga (bucket público de Sen1Floods11 v1.1), particiones oficiales, carga y Dataset."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
BASE = "v1.1/data/flood_events/"
SPLIT_DIR = "v1.1/splits/flood_handlabeled/"
S1H, LBL, S2H = BASE + "HandLabeled/S1Hand/", BASE + "HandLabeled/LabelHand/", BASE + "HandLabeled/S2Hand/"
WS1, WLBL = BASE + "WeaklyLabeled/S1Weak/", BASE + "WeaklyLabeled/S1OtsuLabelWeak/"
SPLIT_NAMES = ["train", "valid", "test", "bolivia"]


def gcs_url(name):
    return f"https://storage.googleapis.com/{BUCKET}/{name}"


def list_remote(prefix, data_root):
    """Lista objetos del bucket público (con caché local)."""
    cache = Path(data_root) / "listing_cache.json"
    cd = json.loads(cache.read_text()) if cache.exists() else {}
    if prefix in cd:
        return cd[prefix]
    names, token = [], None
    while True:
        params = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        names += [it["name"] for it in js.get("items", [])]
        token = js.get("nextPageToken")
        if not token:
            break
    cd[prefix] = names
    cache.write_text(json.dumps(cd))
    return names


def _dl(name, data_root):
    dest = Path(data_root) / name
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(4):
        try:
            with requests.get(gcs_url(name), stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = Path(str(dest) + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as e:  # reintento con espera creciente
            err = e
            time.sleep(2 * (attempt + 1))
    print("FALLÓ la descarga:", name, err)
    return False


def download_many(names, data_root, desc="descarga", workers=16):
    with ThreadPoolExecutor(workers) as ex:
        ok = list(tqdm(ex.map(lambda n: _dl(n, data_root), names), total=len(names), desc=desc))
    assert all(ok), "Hubo descargas fallidas: reintenta o revisa la conexión."


def load_official_splits(data_root):
    """Descarga y lee los CSV oficiales train/valid/test/bolivia."""
    download_many([f"{SPLIT_DIR}flood_{k}_data.csv" for k in SPLIT_NAMES], data_root, "particiones")
    spl = {k: pd.read_csv(Path(data_root) / f"{SPLIT_DIR}flood_{k}_data.csv", header=None, names=["s1", "label"])
           for k in SPLIT_NAMES}
    return {k: v.assign(key=v.s1.str.replace("_S1Hand.tif", "", regex=False),
                        event=lambda d: d.key.str.split("_").str[0]) for k, v in spl.items()}


def select_weak(spl, data_root, n_weak, seed):
    """Submuestra de chips débiles (Otsu). Excluye Bolivia y cualquier chip dorado de valid/test/bolivia."""
    names = [n.split("/")[-1] for n in list_remote(WLBL, data_root) if n.endswith(".tif")]
    weak = pd.DataFrame({"label": names})
    weak["key"] = weak.label.str.replace("_S1OtsuLabelWeak.tif", "", regex=False)
    weak["s1"] = weak.key + "_S1Weak.tif"
    weak["event"] = weak.key.str.split("_").str[0]
    held = set(pd.concat([spl["valid"], spl["test"], spl["bolivia"]]).key)
    weak = weak[(weak.event != "Bolivia") & (~weak.key.isin(held))]
    return weak.sample(n=min(n_weak, len(weak)), random_state=seed).reset_index(drop=True)


class Split:
    """Chips en RAM: X (C,512,512) float16 en dB con NaN, Y (512,512) int8 en {-1,0,1}."""

    def __init__(self, X, Y, keys):
        self.X, self.Y, self.keys = X, Y, list(keys)
        self.events = [k.split("_")[0] for k in self.keys]

    def __len__(self):
        return len(self.X)


def fit512(x, y):
    H, W = y.shape
    if (H, W) == (512, 512):
        return x, y
    px = np.full((x.shape[0], 512, 512), np.nan, np.float16)
    py = np.full((512, 512), -1, np.int8)
    h, w = min(H, 512), min(W, 512)
    px[:, :h, :w] = x[:, :h, :w]
    py[:h, :w] = y[:h, :w]
    return px, py


def load_chip(s1_path, lbl_path):
    import rasterio
    with rasterio.open(s1_path) as a:
        x = a.read().astype(np.float16)
    with rasterio.open(lbl_path) as b:
        y = b.read(1)
    assert x.shape[0] == 2, f"Se esperaban 2 bandas (VV, VH) y hay {x.shape[0]}"
    y = np.where(np.isin(y, (0, 1)), y, -1).astype(np.int8)  # 1 agua, 0 no agua, -1 sin dato
    return fit512(x, y)


def load_split(df, s1_dir, lbl_dir, data_root, desc):
    def one(i):
        r = df.iloc[i]
        return load_chip(Path(data_root) / (s1_dir + r.s1), Path(data_root) / (lbl_dir + r.label))
    with ThreadPoolExecutor(8) as ex:
        res = list(tqdm(ex.map(one, range(len(df))), total=len(df), desc=desc))
    return Split([r[0] for r in res], [r[1] for r in res], df.key.tolist())


def channel_stats(X, n=80):
    """Media/desv. por canal (dB) sobre una submuestra del train dorado."""
    vals = [[] for _ in range(X[0].shape[0])]
    for x in X[:n]:
        for c in range(x.shape[0]):
            v = x[c].astype(np.float32).ravel()
            vals[c].append(v[np.isfinite(v)][::7])
    m = np.array([np.concatenate(v).mean() for v in vals], np.float32)
    s = np.array([np.concatenate(v).std() for v in vals], np.float32)
    return m, s


def make_dataset_class():
    """Define ChipDS aquí para no importar torch al usar solo la descarga/estadística."""
    from torch.utils.data import Dataset
    import torch

    class ChipDS(Dataset):
        def __init__(self, X, Y, mean, std, train=False, crop=256):
            self.X, self.Y, self.train, self.crop = X, Y, train, crop
            self.mean = np.asarray(mean, np.float32).reshape(-1, 1, 1)
            self.std = np.asarray(std, np.float32).reshape(-1, 1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x, y = self.X[i], self.Y[i]
            if self.train:  # recorte aleatorio + flips + rotación de 90°
                c = self.crop
                r = np.random.randint(0, 512 - c + 1)
                s = np.random.randint(0, 512 - c + 1)
                x, y = x[:, r:r + c, s:s + c], y[r:r + c, s:s + c]
                if np.random.rand() < .5:
                    x, y = x[:, :, ::-1], y[:, ::-1]
                if np.random.rand() < .5:
                    x, y = x[:, ::-1, :], y[::-1, :]
                k = np.random.randint(4)
                x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            x = (x.astype(np.float32) - self.mean) / self.std
            x = np.clip(np.nan_to_num(x, nan=0.0), -10, 10)
            return (torch.from_numpy(np.ascontiguousarray(x)),
                    torch.from_numpy(np.ascontiguousarray(y).astype(np.float32)))

    return ChipDS
