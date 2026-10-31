"""Reproduce de extremo a extremo: ablación (§5), test (§6), Bolivia (§7) y firma de errores (§8).

Uso (desde la raíz del repositorio):
    python -m src.run_experiments --mode quick            # ~25 min en una T4
    python -m src.run_experiments --mode full --out outputs_full
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import GRUPO, INTEGRANTES, SECCION, get_device, make_cfg, make_dirs, seed_everything
from .data import (LBL, S1H, SPLIT_NAMES, WLBL, WS1, channel_stats, download_many, load_official_splits,
                   load_split, select_weak)
from .metrics import boot_ci, water_fraction
from .train import train_generic


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "per_chip"}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--data-root", default="data_sen1floods11")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = make_cfg(args.mode)
    data_root, out, _ = make_dirs(args.data_root, args.out)
    device = get_device()
    seed_everything(cfg["seed"])
    print("Dispositivo:", device, "| modo:", args.mode, "|", cfg)

    # ------------------------------ datos ------------------------------
    spl = load_official_splits(data_root)
    gold_all = pd.concat([v.assign(split=k) for k, v in spl.items()], ignore_index=True)
    download_many([S1H + s for s in gold_all.s1] + [LBL + s for s in gold_all.label], data_root, "chips dorados")
    weak = select_weak(spl, data_root, cfg["n_weak"], cfg["seed"])
    download_many([WS1 + s for s in weak.s1] + [WLBL + s for s in weak.label], data_root, "chips débiles")

    D = {k: load_split(spl[k], S1H, LBL, data_root, f"cargando {k}") for k in SPLIT_NAMES}
    W = load_split(weak, WS1, WLBL, data_root, "cargando débiles")
    G = D["train"]
    mean, std = channel_stats(G.X)

    # Control de fuga: Bolivia nunca entra al entrenamiento
    assert not any(e == "Bolivia" for e in G.events + W.events), "Bolivia en entrenamiento"
    assert not (set(G.keys) & set(D["bolivia"].keys)), "chips repetidos entre train y Bolivia"

    wf = water_fraction(G.Y)
    pos_weight = float(min((1 - wf) / wf, 10.0))
    print(f"fracción de agua (train dorado) = {wf:.3f} -> pos_weight = {pos_weight:.2f}")

    evals = {k: (D[k].X, D[k].Y, D[k].keys) for k in ("valid", "test", "bolivia")}
    val = (D["valid"].X, D["valid"].Y)

    # ------------------------------ ablación ------------------------------
    runs = {}

    def phases_for(regime, iters, lr):
        if regime == "G":
            return [(G.X, G.Y, iters, lr)]
        if regime == "G+W":
            return [(G.X + W.X, G.Y + W.Y, iters, lr)]
        if regime == "W→G":
            return [(W.X, W.Y, iters // 2, lr), (G.X, G.Y, iters // 2, lr * 0.3)]

    def run(regime, loss_name):
        name = f"{regime}|{loss_name}"
        if name not in runs:
            runs[name] = train_generic(name, 2, mean, std, phases_for(regime, cfg["iters"], cfg["lr"]),
                                       loss_name, val, evals, cfg, device, pos_weight)
            runs[name].update(regime=regime, loss=loss_name)
        return runs[name]

    for reg in ("G", "G+W", "W→G"):                      # (a) supervisión, pérdida combinada fija
        run(reg, "combined")
    best_regime = max(("G", "G+W", "W→G"), key=lambda r: runs[f"{r}|combined"]["best_val_iou"])
    for ln in ("bce", "dice"):                           # (b) pérdida con la mejor supervisión
        run(best_regime, ln)
    best_name = max(runs, key=lambda n: runs[n]["best_val_iou"])  # selección SOLO por validación
    print(">> mejor estrategia:", best_regime, "| mejor configuración:", best_name)

    rows = []
    for n, r in runs.items():
        t, b, v = r["metrics"]["test"], r["metrics"]["bolivia"], r["metrics"]["valid"]
        rows.append({"Experimento": n, "Supervisión": r["regime"], "Pérdida": r["loss"], "Val IoU": v["iou"],
                     "Test IoU": t["iou"], "Test F1": t["f1"], "Test Prec": t["precision"],
                     "Test Recall": t["recall"], "Bolivia IoU": b["iou"], "Minutos": r["minutes"]})
    abl = pd.DataFrame(rows).set_index("Experimento")
    abl.round(4).to_csv(out / "ablacion.csv")
    pd.concat([r["history"] for r in runs.values()]).to_csv(out / "historial_entrenamiento.csv", index=False)

    # ------------------------------ test y Bolivia ------------------------------
    final = runs[best_name]
    mt, mb = final["metrics"]["test"], final["metrics"]["bolivia"]
    ci_t, ci_b = boot_ci(mt["per_chip"]["iou"]), boot_ci(mb["per_chip"]["iou"])
    print(f"TEST    IoU {mt['iou']:.4f} IC95% [{ci_t[0]:.3f},{ci_t[1]:.3f}] | F1 {mt['f1']:.4f}")
    print(f"BOLIVIA IoU {mb['iou']:.4f} IC95% [{ci_b[0]:.3f},{ci_b[1]:.3f}] | F1 {mb['f1']:.4f}")

    ev = mt["per_chip"].assign(evento=[k.split("_")[0] for k in mt["per_chip"].key])
    ev.groupby("evento")[["iou", "f1", "precision", "recall"]].mean().round(3).to_csv(out / "test_por_evento.csv")
    mb["per_chip"][["key", "iou", "precision", "recall", "f1"]].round(3).to_csv(out / "bolivia_por_chip.csv", index=False)
    gap = [{"Métrica": lab, "En distribución (test)": mt[k], "Bolivia": mb[k], "Δ absoluto": mb[k] - mt[k]}
           for k, lab in [("iou", "IoU"), ("f1", "F1"), ("precision", "Precisión"), ("recall", "Recall"),
                          ("omission", "Omisión (FN)"), ("commission", "Comisión (FP)")]]
    pd.DataFrame(gap).set_index("Métrica").round(4).to_csv(out / "brecha_bolivia.csv")

    # ------------------------------ firma de errores (test + Bolivia) ------------------------------
    stat = {c: dict(n=0, vv=0.0, vh=0.0) for c in ("TP", "TN", "FP", "FN")}
    fn_bright = fn_total = fp_dark = fp_total = 0
    for k in ("test", "bolivia"):
        for x, y, p in zip(D[k].X, D[k].Y, final["probs"][k]):
            valid = (y >= 0) & np.isfinite(x[0]) & np.isfinite(x[1])
            gt, pr = (y == 1) & valid, (p > cfg["thr"]) & valid
            cat = {"TP": pr & gt, "FP": pr & ~gt & valid, "FN": ~pr & gt, "TN": ~pr & ~gt & valid}
            vv, vh = x[0].astype(np.float32), x[1].astype(np.float32)
            for c, m in cat.items():
                stat[c]["n"] += int(m.sum()); stat[c]["vv"] += float(vv[m].sum()); stat[c]["vh"] += float(vh[m].sum())
            fn_bright += int((cat["FN"] & (vv > -12)).sum()); fn_total += int(cat["FN"].sum())
            fp_dark += int((cat["FP"] & (vv < -18)).sum()); fp_total += int(cat["FP"].sum())
    sig = pd.DataFrame({c: dict(píxeles=s["n"], VV_medio_dB=s["vv"] / max(s["n"], 1),
                                VH_medio_dB=s["vh"] / max(s["n"], 1)) for c, s in stat.items()}).T
    sig.round(2).to_csv(out / "firma_errores.csv")

    # ------------------------------ exportación ------------------------------
    summary = dict(integrantes=INTEGRANTES, grupo=GRUPO, seccion=SECCION, modo=args.mode, config=cfg,
                   pos_weight=pos_weight, mejor_estrategia=best_regime, modelo_final=best_name,
                   n_chips={k: len(v) for k, v in D.items()} | {"debiles_usados": len(W)},
                   ablacion=abl.round(4).to_dict(orient="index"), test=_clean(mt), bolivia=_clean(mb),
                   ic95_iou=dict(test=ci_t, bolivia=ci_b),
                   fn_brillantes_pct=100 * fn_bright / max(fn_total, 1),
                   fp_oscuros_pct=100 * fp_dark / max(fp_total, 1))
    (out / "results.json").write_text(json.dumps(_clean(summary), indent=2, ensure_ascii=False, default=str))
    torch.save(final["model"].state_dict(), out / "unet_final.pt")
    print("Guardado en", out.resolve())


if __name__ == "__main__":
    main()
