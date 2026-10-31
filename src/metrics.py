"""Métricas de la clase 'agua' (solo numpy/pandas; sin dependencia de torch)."""
import numpy as np
import pandas as pd


def agg_metrics(probs, Ys, thr=0.5, keys=None):
    """IoU/F1/precisión/recall como media por chip (criterio de los autores del dataset)
    y acumulados sobre píxeles (pooled). Los píxeles con etiqueta -1 se ignoran."""
    per, TP = [], [0, 0, 0, 0]
    for p, y in zip(probs, Ys):
        valid = y >= 0
        gt = (y == 1)
        pr = (p > thr) & valid
        tp = int((pr & gt).sum()); fp = int((pr & ~gt & valid).sum()); fn = int((~pr & gt).sum())
        tn = int(valid.sum()) - tp - fp - fn
        for j, v in enumerate((tp, fp, fn, tn)):
            TP[j] += v
        u = tp + fp + fn
        per.append(dict(iou=tp / u if u else np.nan,
                        precision=tp / (tp + fp) if tp + fp else np.nan,
                        recall=tp / (tp + fn) if tp + fn else np.nan,
                        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                        tp=tp, fp=fp, fn=fn, tn=tn))
    df = pd.DataFrame(per)
    if keys is not None:
        df.insert(0, "key", list(keys))
    tp, fp, fn, tn = TP
    out = {k: float(np.nanmean(df[k])) for k in ("iou", "f1", "precision", "recall")}
    out.update(pooled_iou=tp / max(tp + fp + fn, 1), pooled_f1=2 * tp / max(2 * tp + fp + fn, 1),
               omission=fn / max(tp + fn, 1), commission=fp / max(tp + fp, 1),
               cm=[[tn, fp], [fn, tp]], n_chips=len(df), per_chip=df)
    return out


def boot_ci(v, n=2000, seed=0):
    """IC 95 % bootstrap de la media."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def water_fraction(Ys):
    w = sum(int((y == 1).sum()) for y in Ys)
    v = sum(int((y >= 0).sum()) for y in Ys)
    return w / max(v, 1)
