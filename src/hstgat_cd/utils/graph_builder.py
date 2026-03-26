from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_regression


def symmetric_mi_matrix(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            xi = x[:, i]
            xj = x[:, j]
            valid = ~(np.isnan(xi) | np.isnan(xj))
            if valid.sum() < 10:
                score = 0.0
            else:
                score1 = mutual_info_regression(xi[valid, None], xj[valid], random_state=42)[0]
                score2 = mutual_info_regression(xj[valid, None], xi[valid], random_state=42)[0]
                score = float((score1 + score2) / 2.0)
            out[i, j] = out[j, i] = score
    if out.max() > 0:
        out = out / out.max()
    return out


def subsystem_descriptor(x_sub: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x_sub)
    std = np.nanstd(x_sub)
    vmin = np.nanmin(x_sub)
    vmax = np.nanmax(x_sub)
    vrange = vmax - vmin

    collapsed = np.nanmean(x_sub, axis=1)
    collapsed = np.nan_to_num(collapsed, nan=0.0)
    fft = np.fft.rfft(collapsed)
    amp = np.abs(fft)
    freqs = np.fft.rfftfreq(len(collapsed), d=1.0)
    top_idx = np.argsort(amp)[-3:]
    top_freqs = freqs[top_idx]
    top_amps = amp[top_idx]

    zcr = np.mean((collapsed[:-1] * collapsed[1:]) < 0) if len(collapsed) > 1 else 0.0
    energy = np.mean(collapsed ** 2)
    sk = float(skew(collapsed, nan_policy='omit')) if len(collapsed) > 2 else 0.0
    ku = float(kurtosis(collapsed, nan_policy='omit')) if len(collapsed) > 3 else 0.0
    descriptor = np.array([mean, std, vrange, vmax, vmin, *top_freqs, *top_amps, zcr, energy, sk, ku], dtype=np.float32)
    descriptor = np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
    return descriptor


def build_hierarchical_graph(values: np.ndarray, feature_names: list[str], subsystem_map: dict, config: dict):
    # values: [T, N]
    local_thr = config['graph']['local_threshold']
    global_thr = config['graph']['global_threshold']
    cross_thr = config['graph']['cross_threshold']
    include_self_loops = config['graph'].get('include_self_loops', True)
    top_k_global = config['graph'].get('top_k_global', 3)

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    subsystem_indices = []
    subsystem_names = []
    for name, cols in subsystem_map['subsystems'].items():
        idx = [name_to_idx[c] for c in cols if c in name_to_idx]
        if idx:
            subsystem_indices.append(idx)
            subsystem_names.append(name)

    N = len(feature_names)
    A_local = np.zeros((N, N), dtype=np.float32)
    descriptors = []

    for idxs in subsystem_indices:
        sub = values[:, idxs]
        mi = symmetric_mi_matrix(sub)
        mask = (mi >= local_thr).astype(np.float32)
        if include_self_loops:
            np.fill_diagonal(mask, 1.0)
        for a, ga in enumerate(idxs):
            for b, gb in enumerate(idxs):
                A_local[ga, gb] = mask[a, b]
        descriptors.append(subsystem_descriptor(sub))

    descriptors = np.stack(descriptors, axis=0)
    sub_mi = symmetric_mi_matrix(descriptors)
    A_global_sub = (sub_mi >= global_thr).astype(np.float32)
    if include_self_loops:
        np.fill_diagonal(A_global_sub, 1.0)

    if top_k_global > 0:
        for i in range(A_global_sub.shape[0]):
            order = np.argsort(sub_mi[i])[::-1]
            keep = set(order[: top_k_global + 1])
            for j in range(A_global_sub.shape[1]):
                if j not in keep and i != j:
                    A_global_sub[i, j] = 0.0

    A_cross = np.zeros((N, N), dtype=np.float32)
    for si, idx_i in enumerate(subsystem_indices):
        for sj, idx_j in enumerate(subsystem_indices):
            if si == sj or A_global_sub[si, sj] == 0:
                continue
            pair_data = values[:, idx_i + idx_j]
            pair_mi = symmetric_mi_matrix(pair_data)
            ni = len(idx_i)
            for a, ga in enumerate(idx_i):
                for b, gb in enumerate(idx_j):
                    score = pair_mi[a, ni + b]
                    if score >= cross_thr:
                        A_cross[ga, gb] = 1.0
                        A_cross[gb, ga] = 1.0

    A_global_nodes = np.zeros((N, N), dtype=np.float32)
    for si, idx_i in enumerate(subsystem_indices):
        for sj, idx_j in enumerate(subsystem_indices):
            if A_global_sub[si, sj] == 0:
                continue
            for ga in idx_i:
                for gb in idx_j:
                    A_global_nodes[ga, gb] = 1.0
    if include_self_loops:
        np.fill_diagonal(A_global_nodes, 1.0)
        np.fill_diagonal(A_cross, 1.0)
        np.fill_diagonal(A_local, 1.0)

    return {
        'A_local': A_local,
        'A_global_sub': A_global_sub,
        'A_global_nodes': A_global_nodes,
        'A_cross': A_cross,
        'subsystem_indices': subsystem_indices,
        'subsystem_names': subsystem_names,
    }
