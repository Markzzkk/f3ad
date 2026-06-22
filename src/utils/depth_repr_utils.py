from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _imagenet_normalize_chw(t: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)[:, None, None]
    std = torch.as_tensor(IMAGENET_STD, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean) / std


def _robust_minmax_01(
    x: np.ndarray,
    mask: np.ndarray | None = None,
    lo: float = 1.0,
    hi: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    if mask is not None:
        vals = x[mask]
    else:
        vals = x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(vals, [lo, hi]).astype(np.float32)
    if float(p_hi - p_lo) < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x.astype(np.float32) - p_lo) / (p_hi - p_lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _normalize_depth_01(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if np.issubdtype(d.dtype, np.integer):
        info = np.iinfo(d.dtype)
        d = d.astype(np.float32)
        if info.max > 0:
            d = d / float(info.max)
        else:
            d = _robust_minmax_01(d)
        return d
    d = d.astype(np.float32, copy=False)
    return _robust_minmax_01(d)


def _infer_xyz_order(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 xyz array, got {xyz.shape}")
    H, W, _ = xyz.shape

    neg_fracs = []
    medians = []
    for c in range(3):
        v = xyz[..., c]
        neg_fracs.append(float((v < 0).mean()))
        finite = v[np.isfinite(v)]
        medians.append(float(np.median(finite)) if finite.size else 0.0)
    z_idx = int(np.lexsort(([-m for m in medians], neg_fracs))[0])

    other = [i for i in range(3) if i != z_idx]

    ys = np.linspace(0, H - 1, num=min(64, H), dtype=np.int32)
    xs = np.linspace(0, W - 1, num=min(64, W), dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    samp = (yy, xx)

    def _corr(a, b):
        a = a.reshape(-1).astype(np.float32)
        b = b.reshape(-1).astype(np.float32)
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()) + 1e-6)
        return float((a * b).mean() / denom)

    c0, c1 = other
    v0 = xyz[..., c0][samp]
    v1 = xyz[..., c1][samp]
    corr0x = abs(_corr(v0, xx))
    corr0y = abs(_corr(v0, yy))
    corr1x = abs(_corr(v1, xx))
    corr1y = abs(_corr(v1, yy))

    if corr0x + corr1y >= corr1x + corr0y:
        x_idx, y_idx = c0, c1
    else:
        x_idx, y_idx = c1, c0

    return np.stack([xyz[..., x_idx], xyz[..., y_idx], xyz[..., z_idx]], axis=2).astype(np.float32, copy=False)


def _depth_to_pseudo_xyz(depth01: np.ndarray) -> np.ndarray:
    d = _normalize_depth_01(depth01)
    H, W = d.shape
    fx = fy = float(max(H, W))
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    u = np.arange(W, dtype=np.float32)[None, :]
    v = np.arange(H, dtype=np.float32)[:, None]
    Z = d.astype(np.float32)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.stack([X, Y, Z], axis=2).astype(np.float32)


def _xyz_valid_mask(xyz: np.ndarray) -> np.ndarray:
    z = xyz[..., 2]
    return np.isfinite(z) & (z > 1e-6)


def _xyz_to_normal(xyz_in: np.ndarray) -> np.ndarray:
    xyz = _infer_xyz_order(xyz_in) if (xyz_in.ndim == 3 and xyz_in.shape[2] == 3) else xyz_in.astype(np.float32)
    valid = _xyz_valid_mask(xyz)

    dx = np.zeros_like(xyz, dtype=np.float32)
    dy = np.zeros_like(xyz, dtype=np.float32)
    dx[:, 1:-1, :] = xyz[:, 2:, :] - xyz[:, :-2, :]
    dx[:, 0, :] = xyz[:, 1, :] - xyz[:, 0, :]
    dx[:, -1, :] = xyz[:, -1, :] - xyz[:, -2, :]

    dy[1:-1, :, :] = xyz[2:, :, :] - xyz[:-2, :, :]
    dy[0, :, :] = xyz[1, :, :] - xyz[0, :, :]
    dy[-1, :, :] = xyz[-1, :, :] - xyz[-2, :, :]

    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=2, keepdims=True).astype(np.float32)
    n = n / (norm + 1e-6)
    n01 = (n + 1.0) / 2.0
    neutral = np.array([0.5, 0.5, 1.0], dtype=np.float32)[None, None, :]
    n01 = np.where(valid[:, :, None], n01, neutral)
    return np.clip(n01, 0.0, 1.0).astype(np.float32)


def _xyz_to_hha(xyz_in: np.ndarray) -> np.ndarray:
    xyz = _infer_xyz_order(xyz_in) if (xyz_in.ndim == 3 and xyz_in.shape[2] == 3) else xyz_in.astype(np.float32)
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    valid = _xyz_valid_mask(xyz)

    disp = np.zeros_like(Z, dtype=np.float32)
    disp[valid] = 1.0 / np.maximum(Z[valid], 1e-6)
    D = _robust_minmax_01(disp, mask=valid)

    y_valid = Y[valid]
    if y_valid.size > 0:
        y_top = float(np.max(y_valid))
        Hm = (y_top - Y).astype(np.float32)
        Hc = _robust_minmax_01(Hm, mask=valid)
    else:
        Hc = np.zeros_like(Z, dtype=np.float32)

    n01 = _xyz_to_normal(xyz)
    n = n01 * 2.0 - 1.0
    gravity = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :]
    cosang = np.sum(n * gravity, axis=2)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang).astype(np.float32)
    A = np.clip(ang / (np.pi / 2.0), 0.0, 1.0)

    hha = np.stack([D, Hc, A], axis=2)
    hha[~valid] = 0.0
    return hha.astype(np.float32)


def _tensor_chw_to_hwc_numpy(t: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    x = t.detach().cpu().float()
    if x.ndim == 2:
        return x.numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected CHW or HW tensor, got shape={tuple(x.shape)}")
    return x.permute(1, 2, 0).contiguous().numpy()


def raw_depth_tensor_to_single_channel_depth(depth_t: torch.Tensor) -> torch.Tensor:
    arr = _tensor_chw_to_hwc_numpy(depth_t)

    if arr.ndim == 2:
        z = arr.astype(np.float32, copy=False)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        z = arr[..., 0].astype(np.float32, copy=False)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        xyz = _infer_xyz_order(arr.astype(np.float32, copy=False))
        z = xyz[..., 2]
    else:
        raise ValueError(f"Unsupported raw depth shape for single-depth conversion: {arr.shape}")

    z = np.nan_to_num(z.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(np.ascontiguousarray(z)).unsqueeze(0).float()


def raw_depth_tensor_to_model_repr(
    depth_t: torch.Tensor,
    depth_repr: str = "raw",
    raw_as_single_depth: bool = False,
) -> torch.Tensor:
    dr = str(depth_repr).lower()

    if dr == "raw":
        if raw_as_single_depth:
            return raw_depth_tensor_to_single_channel_depth(depth_t)
        return depth_t.float()

    arr = _tensor_chw_to_hwc_numpy(depth_t)

    if arr.ndim == 2:
        xyz = _depth_to_pseudo_xyz(arr)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        xyz = _depth_to_pseudo_xyz(arr[..., 0])
    elif arr.ndim == 3 and arr.shape[2] == 3:
        xyz = _infer_xyz_order(arr.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported raw depth shape for repr conversion: {arr.shape}")

    if dr == "hha":
        rep = _xyz_to_hha(xyz)
    elif dr in ("normal", "surface_normal", "surface normal"):
        rep = _xyz_to_normal(xyz)
    else:
        raise ValueError(f"Unknown depth_repr: {depth_repr}")

    t = torch.from_numpy(np.ascontiguousarray(rep)).permute(2, 0, 1).contiguous().float()
    return _imagenet_normalize_chw(t)


def raw_depth_batch_to_model_repr(
    depths: torch.Tensor,
    depth_repr: str = "raw",
    raw_as_single_depth: bool = False,
) -> torch.Tensor | None:
    if depths is None:
        return None

    if depths.ndim == 3:
        return raw_depth_tensor_to_model_repr(
            depths, depth_repr=depth_repr, raw_as_single_depth=raw_as_single_depth
        )

    if depths.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(depths.shape)}")

    device = depths.device
    dtype = depths.dtype if torch.is_floating_point(depths) else torch.float32
    outs = [
        raw_depth_tensor_to_model_repr(
            depths[i], depth_repr=depth_repr, raw_as_single_depth=raw_as_single_depth
        )
        for i in range(depths.shape[0])
    ]
    return torch.stack(outs, dim=0).to(device=device, dtype=dtype, non_blocking=True)


def raw_depth_batch_to_single_channel_depth(depths: torch.Tensor) -> torch.Tensor | None:
    if depths is None:
        return None

    if depths.ndim == 3:
        return raw_depth_tensor_to_single_channel_depth(depths)

    if depths.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(depths.shape)}")

    device = depths.device
    outs = [raw_depth_tensor_to_single_channel_depth(depths[i]) for i in range(depths.shape[0])]
    return torch.stack(outs, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
