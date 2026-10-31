"""Pérdidas que ENMASCARAN los píxeles sin dato (etiqueta -1)."""
import torch
import torch.nn.functional as F


def masked_bce(logits, t, valid, pos_weight=None):
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    l = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pw)
    return (l * valid).sum() / valid.sum().clamp(min=1)


def masked_dice(logits, t, valid, eps=1e-6):
    p = torch.sigmoid(logits) * valid
    tt = t * valid
    return 1 - (2 * (p * tt).sum() + eps) / (p.sum() + tt.sum() + eps)


def make_loss(name, pos_weight=None):
    """name: 'bce' | 'dice' | 'combined' (BCE ponderada + Dice)."""
    if name == "bce":
        return lambda lg, t, v: masked_bce(lg, t, v)
    if name == "dice":
        return lambda lg, t, v: masked_dice(lg, t, v)
    if name == "combined":
        return lambda lg, t, v: masked_bce(lg, t, v, pos_weight) + masked_dice(lg, t, v)
    raise ValueError(name)
