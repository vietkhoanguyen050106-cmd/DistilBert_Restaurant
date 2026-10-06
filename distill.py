"""Các hàm loss cho distillation."""
import torch
import torch.nn.functional as F


def logit_kd_loss(s_logits, t_logits, temp: float):
    return F.kl_div(
        F.log_softmax(s_logits.float() / temp, dim=-1),
        F.softmax(t_logits.float() / temp, dim=-1),
        reduction="batchmean",
    ) * temp ** 2


def hidden_distill_loss(s_hs, t_hs, mask, s_idx, t_idx):
    """MSE giữa hidden states student/teacher (LayerNorm không tham số để cùng scale),
    chỉ tính trên token thật (bỏ padding). Student và teacher cùng hidden size (768)."""
    m = mask.unsqueeze(-1).float()
    denom = m.sum() * s_hs[0].size(-1)
    loss = 0.0
    for si, ti in zip(s_idx, t_idx):
        s = F.layer_norm(s_hs[si].float(), s_hs[si].shape[-1:])
        t = F.layer_norm(t_hs[ti].float(), t_hs[ti].shape[-1:])
        loss = loss + (((s - t) ** 2) * m).sum() / denom
    return loss / len(s_idx)
