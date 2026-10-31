"""Entrenamiento genérico por fases y predicción (precisión mixta, AdamW + warmup/coseno)."""
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import seed_everything
from .data import make_dataset_class
from .losses import make_loss
from .metrics import agg_metrics
from .model import build_model

ChipDS = make_dataset_class()


def _winit(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


@torch.no_grad()
def predict(model, S_X, mean, std, device, bs=8):
    use_amp = device.type == "cuda"
    model.eval()
    ds = ChipDS(S_X, [np.zeros((512, 512), np.int8)] * len(S_X), mean, std)
    out = []
    for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = model(x.to(device))
        out.append(torch.sigmoid(lg.float()).squeeze(1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def train_generic(name, in_ch, mean, std, phases, loss_name, val, evals, cfg, device,
                  pos_weight=None, verbose=True):
    """
    phases: lista de (X, Y, iters, lr), entrenadas en secuencia (p. ej. preentrenamiento débil y luego ajuste dorado).
    val   : (X, Y) para elegir el mejor punto (IoU medio por chip).
    evals : {nombre: (X, Y, keys)} evaluados al final con el mejor punto.
    """
    use_amp = device.type == "cuda"
    seed_everything(cfg["seed"])
    t0 = time.time()
    model = build_model(in_ch, cfg["base"], cfg["depth"], device)
    loss_fn = make_loss(loss_name, pos_weight)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:  # PyTorch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best = dict(iou=-1, state=None, step=None)
    hist, step_total = [], 0
    for pi, (X, Y, iters, lr) in enumerate(phases):
        ds = ChipDS(X, Y, mean, std, train=True, crop=cfg["crop"])
        bs = min(cfg["batch_size"], len(ds))
        loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) >= bs,
                            num_workers=cfg["workers"], worker_init_fn=_winit,
                            persistent_workers=cfg["workers"] > 0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
        warm = max(1, int(0.05 * iters))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, iters) / iters)))
        it, run_loss, nl = 0, 0.0, 0
        model.train()
        while it < iters:
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device).unsqueeze(1)
                valid = (y >= 0).float()
                t = y.clamp(min=0)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    lg = model(x)
                loss = loss_fn(lg.float(), t, valid)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                it += 1
                step_total += 1
                run_loss += loss.item()
                nl += 1
                if it % cfg["eval_every"] == 0 or it == iters:
                    pv = predict(model, val[0], mean, std, device)
                    m = agg_metrics(pv, val[1], cfg["thr"])
                    hist.append(dict(run=name, fase=pi + 1, iter=step_total, loss=run_loss / nl,
                                     val_iou=m["iou"], val_f1=m["f1"]))
                    if verbose:
                        print(f"[{name}] fase {pi+1} it {it}/{iters} | loss {run_loss/nl:.4f} | "
                              f"val IoU {m['iou']:.4f} F1 {m['f1']:.4f}")
                    run_loss, nl = 0.0, 0
                    if m["iou"] > best["iou"]:
                        best = dict(iou=m["iou"], step=step_total,
                                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    model.train()
                if it >= iters:
                    break
        del loader
    model.load_state_dict(best["state"])
    res = dict(name=name, history=pd.DataFrame(hist), best_val_iou=best["iou"], best_step=best["step"],
               model=model, metrics={}, probs={})
    for k, (Xe, Ye, ke) in evals.items():
        p = predict(model, Xe, mean, std, device)
        res["probs"][k] = p
        res["metrics"][k] = agg_metrics(p, Ye, cfg["thr"], ke)
    res["minutes"] = (time.time() - t0) / 60
    if verbose:
        print(f"[{name}] listo en {res['minutes']:.1f} min | mejor val IoU {best['iou']:.4f} (paso {best['step']})")
    return res
