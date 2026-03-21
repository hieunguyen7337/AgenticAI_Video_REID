from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_tensor(features: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(features, np.ndarray):
        return torch.from_numpy(features)
    return features


def euclidean_distance(query_features: torch.Tensor | np.ndarray, gallery_features: torch.Tensor | np.ndarray) -> np.ndarray:
    qf = _as_tensor(query_features).float()
    gf = _as_tensor(gallery_features).float()
    m, n = qf.shape[0], gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(query_features: torch.Tensor | np.ndarray, gallery_features: torch.Tensor | np.ndarray, normalize: bool = True) -> np.ndarray:
    qf = _as_tensor(query_features).float()
    gf = _as_tensor(gallery_features).float()
    if normalize:
        qf = torch.nn.functional.normalize(qf, dim=1, p=2)
        gf = torch.nn.functional.normalize(gf, dim=1, p=2)
    return qf.mm(gf.t()).cpu().numpy()


def _distance_for_metric(query_features: torch.Tensor | np.ndarray, gallery_features: torch.Tensor | np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return euclidean_distance(query_features, gallery_features)
    if metric == "cosine":
        return 1.0 - cosine_similarity(query_features, gallery_features, normalize=True)
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def rank_gallery(
    query_features: torch.Tensor | np.ndarray,
    gallery_features: torch.Tensor | np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    distances = _distance_for_metric(query_features, gallery_features, metric)
    return np.argsort(distances, axis=1)


def eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(1, tmp_cmc.shape[0] + 1) * 1.0)
        ap = (np.asarray(tmp_cmc) * orig_cmc).sum() / num_rel
        all_ap.append(ap)

    if num_valid_q == 0:
        raise ValueError("All query identities are absent from the gallery after same-camera filtering")

    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_ap))
    return all_cmc, mAP


def evaluate_reid(
    query_features: torch.Tensor | np.ndarray,
    gallery_features: torch.Tensor | np.ndarray,
    query_pids: Any,
    gallery_pids: Any,
    query_camids: Any,
    gallery_camids: Any,
    metric: str = "cosine",
    max_rank: int = 50,
) -> dict[str, Any]:
    distmat = _distance_for_metric(query_features, gallery_features, metric)
    q_pids = np.asarray(query_pids)
    g_pids = np.asarray(gallery_pids)
    q_camids = np.asarray(query_camids)
    g_camids = np.asarray(gallery_camids)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=max_rank)
    return {
        "cmc": cmc,
        "mAP": mAP,
        "rank1": float(cmc[0]),
        "distmat": distmat,
    }
