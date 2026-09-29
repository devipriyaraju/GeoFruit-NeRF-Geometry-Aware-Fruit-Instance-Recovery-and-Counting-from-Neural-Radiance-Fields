from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel_size).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def radius_outlier_removal(pts: np.ndarray, radius: float, min_nb: int) -> np.ndarray:
    if len(pts) == 0:
        return pts
    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, radius, return_length=True)
    return pts[np.asarray(counts) >= min_nb]


def hull_volume(p: np.ndarray) -> float:
    if len(p) < 4:
        return 0.0
    try:
        return float(ConvexHull(p).volume)
    except Exception:
        return 0.0


def main_extent(p: np.ndarray) -> float:
    q = p - p.mean(0)
    if len(q) < 3:
        return 0.0
    axis = np.linalg.svd(q[: min(len(q), 20000)], full_matrices=False)[2][0]
    proj = q @ axis
    return float(np.percentile(proj, 98) - np.percentile(proj, 2))


def fib_sphere(n: int, r: float) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def count_fruits(pts: np.ndarray, h: float, cfg: dict, verbose: bool = True, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    info = {}
    p = voxel_downsample(pts, h)
    p = radius_outlier_removal(p, cfg["outlier_radius_mult"] * h, cfg["outlier_min_nb"])
    info["points_after_filter"] = len(p)
    if len(p) == 0:
        return 0, np.zeros((0, 3)), info, p, np.zeros(0, int)

    labels = DBSCAN(eps=cfg["dbscan_eps_mult"] * h, min_samples=cfg["dbscan_min_samples"]).fit_predict(p)
    ids = [k for k in np.unique(labels) if k >= 0]
    clusters = [p[labels == k] for k in ids]
    if not clusters:
        return 0, np.zeros((0, 3)), info, p, np.full(len(p), -1)

    ext = np.array([main_extent(c) for c in clusters])
    vol = np.array([hull_volume(c) for c in clusters])
    diam = cfg.get("fruit_diameter") or float(np.median(ext[ext > 0]))
    typical = (ext > 0.75 * diam) & (ext < 1.25 * diam)
    v_t = float(np.median(vol[typical])) if typical.any() else float(np.median(vol))
    radius = diam / 2
    info.update(dbscan_clusters=len(clusters), fruit_diameter=diam, template_volume=v_t)

    ratio = vol / max(v_t, 1e-12)
    tiny = np.where(ratio < cfg["min_size_factor"])[0]
    multi = np.where(ratio > cfg["multi_factor"])[0]
    single = np.setdiff1d(np.arange(len(clusters)), np.concatenate([tiny, multi]))
    centers = [clusters[i].mean(0) for i in single]
    final_labels = np.full(len(p), -1)

    kept_tiny = 0
    if len(tiny):
        tc = np.array([clusters[i].mean(0) for i in tiny])
        parent = list(range(len(tiny)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in cKDTree(tc).query_pairs(radius):
            parent[find(a)] = find(b)
        groups = {}
        for a in range(len(tiny)):
            groups.setdefault(find(a), []).append(tiny[a])
        for group in groups.values():
            merged = np.concatenate([clusters[i] for i in group])
            if hull_volume(merged) >= cfg["min_size_factor"] * v_t:
                centers.append(merged.mean(0))
                kept_tiny += 1

    if len(single):
        template_id = single[np.argmin(np.abs(vol[single] - v_t))]
        template = clusters[template_id] - clusters[template_id].mean(0)
        if len(template) > 400:
            template = template[rng.choice(len(template), 400, replace=False)]
    else:
        template = fib_sphere(300, radius)

    if cfg.get("fruit_template") is not None:
        template = np.asarray(cfg["fruit_template"])
        template = template - template.mean(0)

    multi_counts = []
    mode = cfg.get("multi_mode", "hausdorff")
    n_single = float(np.median([len(clusters[j]) for j in single])) if len(single) else None

    for i in multi:
        y = clusters[i]
        if len(y) > 3000:
            y = y[rng.choice(len(y), 3000, replace=False)]

        def centers_for(k):
            if k == 1:
                return [y.mean(0)]
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(y)
            return [y[lab == j].mean(0) for j in range(k)]

        if mode == "volume" or (mode == "points" and n_single):
            est = vol[i] / max(v_t, 1e-12) if mode == "volume" else len(clusters[i]) / n_single
            k = int(np.clip(round(est), 1, min(cfg["max_fruits_per_cluster"], len(y))))
            centers_for_cluster = centers_for(k)
        else:
            best = (np.inf, 1, None)
            for k in range(1, min(cfg["max_fruits_per_cluster"], len(y)) + 1):
                candidate = centers_for(k)
                x = np.concatenate([template + c for c in candidate])
                d_hd = max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])
                if d_hd < best[0]:
                    best = (d_hd, k, candidate)
            k = best[1]
            centers_for_cluster = best[2]

        multi_counts.append(k)
        centers.extend(centers_for_cluster)

    for j, k in enumerate(ids):
        final_labels[labels == k] = j

    count = len(centers)
    info.update(
        single=len(single),
        tiny_raw=len(tiny),
        tiny_kept=kept_tiny,
        multi=len(multi),
        fruits_in_multi=int(sum(multi_counts)),
        count=count,
    )
    if verbose:
        print(f"filtered pts {info['points_after_filter']:,} | DBSCAN clusters {len(clusters)} | fruit diameter {diam:.4f}")
        print(f"single {len(single)} | multi {len(multi)} -> {info['fruits_in_multi']} fruits | tiny {len(tiny)} -> kept {kept_tiny} | TOTAL = {count}")
    return count, np.asarray(centers), info, p, final_labels


def match_to_gt(pred: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    if len(pred) == 0 or len(gt) == 0:
        return {"tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    dist = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
    rows, cols = linear_sum_assignment(dist)
    tp = int((dist[rows, cols] < thr).sum())
    precision = tp / len(pred)
    recall = tp / len(gt)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "precision": precision, "recall": recall, "f1": f1}
