"""Configuración global, presets de cómputo y utilidades de reproducibilidad."""
import random
from pathlib import Path

import numpy as np

GRUPO, SECCION = "04", "C"
INTEGRANTES = [
    "Anllelo Paolito Carreazo Yrizales (líder)",
    "Salvador Lincoln Cabrera Taco",
    "Pedro Alexander Marquez Lamadrid",
    "Yda Ramirez Rojas",
    "Mijael Aguirre Zegarra",
]

# "quick": corrida reducida (sesión corta de Colab) | "full": más iteraciones y más chips débiles
PRESETS = {
    "quick": dict(iters=600, eval_every=100, n_weak=600),
    "full": dict(iters=3000, eval_every=250, n_weak=2000),
}


def make_cfg(mode: str = "quick") -> dict:
    cfg = dict(PRESETS[mode])
    cfg.update(batch_size=16, crop=256, lr=5e-4, wd=1e-4, base=64, depth=4,
               thr=0.5, seed=42, workers=2)
    return cfg


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs(data_root="data_sen1floods11", out="outputs"):
    data_root, out = Path(data_root), Path(out)
    figs = out / "figs"
    for d in (data_root, out, figs):
        d.mkdir(parents=True, exist_ok=True)
    return data_root, out, figs
