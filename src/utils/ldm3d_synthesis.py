from __future__ import annotations

import os
import io
import re
import glob
import json
import math
import random
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

try:
    import tifffile as tiff
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False


# -----------------------------------------------------------
# Small helpers
# -----------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _atomic_save_pil(pil: Image.Image, out_path: str) -> None:
    tmp = out_path + ".tmp"
    ext = os.path.splitext(out_path)[1].lower()
    fmt = "PNG" if ext == ".png" else "PNG"
    pil.save(tmp, format=fmt)
    os.replace(tmp, out_path)


def _atomic_write_json(obj: dict, out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def _atomic_write_npy(arr: np.ndarray, out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out_path)


def _list_cached_anomalies(d: str) -> List[int]:
    if not os.path.isdir(d):
        return []
    idxs = []
    for fn in os.listdir(d):
        if fn.startswith("anomaly_rgb_") and fn.endswith(".png"):
            try:
                idx = int(fn[len("anomaly_rgb_"):-len(".png")])
                idxs.append(idx)
            except Exception:
                pass
    idxs.sort()
    return idxs


def _next_cache_index(d: str) -> int:
    idxs = _list_cached_anomalies(d)
    return (idxs[-1] + 1) if idxs else 0


def _cache_dir(cache_root: str, dataset: str, cls: str, sample_id: str, defect: str) -> str:
    return os.path.join(cache_root, dataset, cls, sample_id, defect)


def _tensor01_to_pil_rgb(x01: torch.Tensor) -> Image.Image:
    """
    x01: [3,H,W] in [0,1]
    """
    x = (x01.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return Image.fromarray(x, mode="RGB")


def _pil_to_tensor01_rgb(pil: Image.Image) -> torch.Tensor:
    x = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def _mask01_to_pil(mask01: torch.Tensor) -> Image.Image:
    m = (mask01[0].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def _tensor01_depth_to_u16(depth01: torch.Tensor) -> np.ndarray:
    """
    depth01: [1,H,W] in [0,1]
    -> uint16 [H,W]
    """
    d = depth01[0].detach().float().clamp(0, 1).cpu().numpy()
    return (d * 65535.0).round().astype(np.uint16)


def _u16_to_tensor01_depth(arr_u16: np.ndarray) -> torch.Tensor:
    d = arr_u16.astype(np.float32) / 65535.0
    return torch.from_numpy(d)[None]


def load_binary_mask(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        if not HAS_TIFFFILE:
            raise RuntimeError("tifffile is required to read .tif/.tiff masks")
        arr = tiff.imread(path)
        m = (np.asarray(arr) > 0).astype(np.uint8)
    else:
        m = Image.open(path).convert("L")
        m = (np.array(m) > 0).astype(np.uint8)
    return m


def morph_close_bin(mask01: np.ndarray, k: int = 7, iters: int = 2) -> np.ndarray:
    x = torch.from_numpy(mask01.astype(np.float32))[None, None]
    for _ in range(iters):
        dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        er = 1.0 - F.max_pool2d(1.0 - dil, kernel_size=k, stride=1, padding=k // 2)
        x = er
    return (x[0, 0].numpy() > 0.5).astype(np.uint8)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx <= mn + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def read_depth_from_xyz(xyz_path: str) -> np.ndarray:
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile is required to read xyz tiff")
    xyz = tiff.imread(xyz_path)
    xyz = np.asarray(xyz)
    if xyz.ndim < 3 or xyz.shape[-1] < 3:
        raise ValueError(f"Unexpected xyz shape: {xyz.shape} for {xyz_path}")
    z = xyz[..., 2].astype(np.float32)
    if np.isnan(z).any():
        med = float(np.nanmedian(z)) if np.isfinite(np.nanmedian(z)) else 0.0
        z = np.nan_to_num(z, nan=med, posinf=med, neginf=med)
    return z


def compute_object_mask_from_xyz(xyz_path: str, k_close: int = 7) -> np.ndarray:
    z = read_depth_from_xyz(xyz_path)
    z01 = normalize01(z)
    H, W = z01.shape

    p = 60
    th1 = np.percentile(z01, p)
    th2 = np.percentile(z01, 100 - p)
    cand1 = (z01 >= th1).astype(np.uint8)
    cand2 = (z01 <= th2).astype(np.uint8)

    cy1, cy2 = H // 4, 3 * H // 4
    cx1, cx2 = W // 4, 3 * W // 4
    score1 = cand1[cy1:cy2, cx1:cx2].sum()
    score2 = cand2[cy1:cy2, cx1:cx2].sum()
    obj = cand1 if score1 >= score2 else cand2

    obj = morph_close_bin(obj, k=k_close, iters=2)
    return obj


def random_affine_mask(mask01: np.ndarray, angle: float, scale: float, hflip: bool, vflip: bool) -> np.ndarray:
    H, W = mask01.shape
    im = Image.fromarray(mask01.astype(np.uint8) * 255, "L")
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    im = im.resize((newW, newH), resample=Image.NEAREST)
    if hflip:
        im = ImageOps.mirror(im)
    if vflip:
        im = ImageOps.flip(im)
    im = im.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    return (np.array(im) > 0).astype(np.uint8)


def elastic_deform_mask(mask01: np.ndarray, alpha: float = 20.0, sigma: float = 6.0) -> np.ndarray:
    H, W = mask01.shape
    x = torch.from_numpy(mask01.astype(np.float32))[None, None]

    def gauss_1d(ks: int, s: float) -> torch.Tensor:
        ax = torch.arange(ks) - ks // 2
        g = torch.exp(-(ax ** 2) / (2 * s * s))
        return g / g.sum()

    ks = int(max(3, 2 * int(3 * sigma) + 1))
    g = gauss_1d(ks, sigma)
    g2d = torch.outer(g, g)[None, None]

    dx = torch.randn(1, 1, H, W) * alpha
    dy = torch.randn(1, 1, H, W) * alpha
    pad = ks // 2
    dx = F.conv2d(F.pad(dx, (pad, pad, pad, pad), mode="reflect"), g2d)
    dy = F.conv2d(F.pad(dy, (pad, pad, pad, pad), mode="reflect"), g2d)

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)[None]
    dx_n = dx[0, 0] * (2.0 / max(1, W - 1))
    dy_n = dy[0, 0] * (2.0 / max(1, H - 1))
    grid = grid + torch.stack([dx_n, dy_n], dim=-1)[None]

    out = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return (out[0, 0].numpy() > 0.5).astype(np.uint8)


def place_mask_on_canvas(mask_small01: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = mask_small01.shape
    if h <= 0 or w <= 0:
        return np.zeros((H, W), dtype=np.uint8)

    if h >= H or w >= W:
        scale = min((H - 1) / max(1, h), (W - 1) / max(1, w)) * 0.9
        newh = max(1, int(round(h * scale)))
        neww = max(1, int(round(w * scale)))
        im = Image.fromarray(mask_small01.astype(np.uint8) * 255, "L").resize((neww, newh), Image.NEAREST)
        mask_small01 = (np.array(im) > 0).astype(np.uint8)
        h, w = mask_small01.shape

    y = random.randint(0, H - h)
    x = random.randint(0, W - w)
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], mask_small01)
    return canvas


def _eyecandies_mask_nonempty(mask_path: str) -> bool:
    try:
        m = Image.open(mask_path)
        return m.getbbox() is not None
    except Exception:
        return False


def build_mask_pool(
    dataset: str,
    dataset_root: str,
    eyecandies_splits: Sequence[str] = ("train", "val", "test_public"),
    skip_empty_masks: bool = True,
) -> Dict[str, List[str]]:
    """
    Build per-class real-defect mask pool. For EyeCandies, filters all-black masks by default.
    """
    root = Path(dataset_root)
    pool: Dict[str, List[str]] = {}

    if not root.exists():
        warnings.warn(f"[LDM3DAnomalyGenerator] mask pool root not found: {root}")
        return pool

    dataset = dataset.lower()
    if dataset == "mvtec3d":
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            files: List[str] = []
            test_dir = cls_dir / "test"
            if not test_dir.exists():
                continue
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir() or defect_dir.name == "good":
                    continue
                gt_dir = defect_dir / "gt"
                cand_dirs = []
                if gt_dir.exists():
                    cand_dirs.append(gt_dir)
                    rgb_sub = gt_dir / "rgb"
                    if rgb_sub.exists():
                        cand_dirs.append(rgb_sub)
                for d in cand_dirs:
                    for ext in ("*.png", "*.tif", "*.tiff"):
                        files += glob.glob(str(d / ext))
            files = sorted(set(files))
            files = [f for f in files if not skip_empty_masks or (load_binary_mask(f).sum() > 0)]
            if files:
                pool[cls_dir.name] = files

    elif dataset == "eyecandies":
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            files: List[str] = []
            for split in eyecandies_splits:
                data_dir = cls_dir / split / "data"
                if not data_dir.is_dir():
                    continue
                for p in sorted(data_dir.glob("*_mask.png")):
                    if skip_empty_masks and (not _eyecandies_mask_nonempty(str(p))):
                        continue
                    files.append(str(p))
            if files:
                pool[cls_dir.name] = files

    else:
        raise ValueError(f"Unsupported dataset for LDM3D mask pool: {dataset}")

    return pool


# -----------------------------------------------------------
# LDM3D SDEdit anomaly generator (RGBD)
# -----------------------------------------------------------

class LDM3DSDEditAnomalyGenerator:
    """
    RGBD anomaly generator for FoundAD training.

    Interface is intentionally aligned with CutPasteUnion usage in train_rgbd_2_v2.py:
        _, (imgs_abn, depths_abn) = gen(imgs, labels=labels, depths=depths, paths=paths, depth_paths=depth_paths)

    Notes:
      - Prompt is kept generic by default to match your LoRA fine-tuning:
            defect_prompt = "an object with defects"
            clean_prompt  = "a clean object"
      - EyeCandies all-black masks are skipped when building mask pool.
      - Uses cache to avoid repeated slow diffusion generation.
    """

    def __init__(
        self,
        # dataset / masks
        dataset: str,
        dataset_root: str,
        eyecandies_splits_for_masks: Sequence[str] = ("train", "val", "test_public"),
        skip_empty_masks: bool = True,

        # model
        model_path: str = "",
        unet_lora_path: Optional[str] = None,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        target_modules: Sequence[str] = ("to_q", "to_k", "to_v", "to_out.0"),

        # inference / sdedit
        resolution: Optional[int] = None,  # if None, use input H/W
        steps: int = 30,
        strength: float = 0.55,
        guidance: float = 6.0,
        use_clean_as_negative: bool = False,
        defect_prompt: str = "an object with defects",
        clean_prompt: str = "a clean object",

        # dtype / device
        bf16: bool = True,
        fp16: bool = False,
        device: Optional[str] = None,
        local_files_only: bool = False,

        # cache
        cache_root: Optional[str] = None,
        cache_k: int = 32,
        save_original_once: bool = True,
        save_meta: bool = True,

        # mask augment
        use_elastic: bool = True,
        alpha: float = 20.0,
        sigma: float = 6.0,
        scale_min: float = 0.3,
        scale_max: float = 1.0,
        rotate_max: float = 45.0,
        attempts_per_mask: int = 30,
        min_coverage: float = 0.8,
        use_obj_intersection_if_possible: bool = True,

        # input normalization in FoundAD training pipeline
        rgb_input_norm: str = "imagenet",     # imagenet | none
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        depth_input_norm: str = "auto",       # auto | none | m11 | zscore
        depth_mean: float = 0.0,
        depth_std: float = 1.0,

        # fallback behavior
        on_failure: str = "return_clean",     # return_clean | raise
    ):
        dataset = dataset.lower()
        if dataset not in ("mvtec3d", "eyecandies"):
            raise ValueError(f"LDM3DSDEditAnomalyGenerator only supports mvtec3d/eyecandies, got {dataset}")
        self.dataset = dataset
        self.dataset_root = str(dataset_root)
        self.skip_empty_masks = bool(skip_empty_masks)
        self.eyecandies_splits_for_masks = tuple(eyecandies_splits_for_masks)

        self.steps = int(steps)
        self.strength = float(strength)
        self.guidance = float(guidance)
        self.resolution = int(resolution) if resolution is not None else None
        self.use_clean_as_negative = bool(use_clean_as_negative)
        self.defect_prompt = defect_prompt
        self.clean_prompt = clean_prompt

        self.cache_root = cache_root
        self.cache_k = int(cache_k)
        self.save_original_once = bool(save_original_once)
        self.save_meta = bool(save_meta)

        self.use_elastic = bool(use_elastic)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.rotate_max = float(rotate_max)
        self.attempts_per_mask = int(attempts_per_mask)
        self.min_coverage = float(min_coverage)
        self.use_obj_intersection_if_possible = bool(use_obj_intersection_if_possible)

        self.rgb_input_norm = str(rgb_input_norm).lower()
        self.rgb_mean = torch.tensor(list(rgb_mean), dtype=torch.float32).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor(list(rgb_std), dtype=torch.float32).view(1, 3, 1, 1)
        self.depth_input_norm = str(depth_input_norm).lower()
        self.depth_mean = float(depth_mean)
        self.depth_std = float(depth_std)
        self.on_failure = str(on_failure).lower()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if fp16 and bf16:
            warnings.warn("Both bf16 and fp16 set; using bf16.")
            fp16 = False
        if bf16:
            pipe_dtype = torch.bfloat16
        elif fp16:
            pipe_dtype = torch.float16
        else:
            pipe_dtype = torch.float32
        self.pipe_dtype = pipe_dtype

        self.mask_pool = build_mask_pool(
            dataset=self.dataset,
            dataset_root=self.dataset_root,
            eyecandies_splits=self.eyecandies_splits_for_masks,
            skip_empty_masks=self.skip_empty_masks,
        )

        if len(self.mask_pool) == 0:
            warnings.warn(f"[LDM3DAnomalyGenerator] mask pool is empty for dataset={self.dataset} root={self.dataset_root}")

        # lazy heavy imports here to avoid impacting cutpaste path
        from diffusers import StableDiffusionLDM3DPipeline, DDIMScheduler

        self.pipe = StableDiffusionLDM3DPipeline.from_pretrained(
            model_path,
            torch_dtype=self.pipe_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=local_files_only,
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self._set_mem_savers()

        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()

        if unet_lora_path:
            self._load_lora(unet_lora_path, rank=lora_rank, alpha=lora_alpha, target_modules=tuple(target_modules))

        # cache text embeddings (generic prompt)
        self._cond_emb = None
        self._uncond_emb = None
        self._refresh_prompt_embeds()

    def _set_mem_savers(self) -> None:
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            except Exception:
                pass

    def _load_lora(
        self,
        lora_path: str,
        rank: int = 16,
        alpha: int = 16,
        target_modules: Sequence[str] = ("to_q", "to_k", "to_v", "to_out.0"),
    ) -> None:
        """
        Load LoRA weights saved by your train_universal.py:
            state_dict = {k:v for k,v in unet.state_dict().items() if "lora" in k}
            torch.save(state_dict, ".../unet_lora.pth")
        """
        try:
            from peft import LoraConfig
        except Exception as e:
            raise RuntimeError("peft is required for loading LoRA into LDM3D UNet") from e

        cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=list(target_modules),
        )
        adapter_name = "anomaly_adapter"
        self.pipe.unet.add_adapter(cfg, adapter_name=adapter_name)

        sd = torch.load(lora_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "lora" in sd and isinstance(sd["lora"], dict):
            # tolerate wrapped format {"lora": {...}}
            sd = sd["lora"]
        if not isinstance(sd, dict):
            raise ValueError(f"Unsupported LoRA checkpoint format at {lora_path}")

        lora_sd = {k: v for k, v in sd.items() if "lora" in k.lower()}
        if not lora_sd:
            warnings.warn(f"[LDM3DAnomalyGenerator] No LoRA tensors found in {lora_path}; attempting strict=False load on full dict")
            lora_sd = sd

        missing, unexpected = self.pipe.unet.load_state_dict(lora_sd, strict=False)
        if unexpected:
            warnings.warn(f"[LDM3DAnomalyGenerator] LoRA unexpected keys: {len(unexpected)}")
        # missing keys are expected because only LoRA params are loaded

    def _refresh_prompt_embeds(self) -> None:
        self._cond_emb = self._encode_text(self.defect_prompt)
        neg_prompt = self.clean_prompt if self.use_clean_as_negative else ""
        self._uncond_emb = self._encode_text(neg_prompt)

    def _encode_text(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer
        ids = tok(
            prompt,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = self.pipe.text_encoder(
                input_ids=ids.input_ids.to(self.device),
                attention_mask=ids.attention_mask.to(self.device),
            )[0]
        return emb.to(device=self.device, dtype=self.pipe.unet.dtype)

    # ---------------- path / class / id helpers ----------------
    def _infer_cls_from_path(self, p: Path) -> str:
        for part in p.parts:
            if part in self.mask_pool:
                return part
        # fallback heuristics
        if self.dataset == "mvtec3d":
            # /.../<cls>/train/good/rgb/xxx.png
            parts = list(p.parts)
            for i, part in enumerate(parts):
                if part == "train" and i - 1 >= 0:
                    cand = parts[i - 1]
                    if cand in self.mask_pool:
                        return cand
        elif self.dataset == "eyecandies":
            # /.../<cls>/<split>/data/xxx_image_0.png
            parts = list(p.parts)
            if "data" in parts:
                i = parts.index("data")
                if i - 2 >= 0:
                    return parts[i - 2]
        return p.parent.name

    def _infer_sample_id_from_path(self, p: Path) -> str:
        stem = p.stem
        # EyeCandies: 29_image_0 -> 29
        m = re.match(r"(.+)_image_\d+$", stem)
        if m:
            return m.group(1)
        m = re.match(r"(.+)_depth$", stem)
        if m:
            return m.group(1)
        return stem

    def _defect_from_mask_path(self, mask_path: str) -> str:
        p = Path(mask_path)
        parts = p.parts
        if self.dataset == "mvtec3d" and "test" in parts:
            i = parts.index("test")
            if i + 1 < len(parts):
                return parts[i + 1] if parts[i + 1] else "defect"
        if self.dataset == "eyecandies":
            return "defect"
        return p.parent.name if p.parent.name else "defect"

    def _infer_xyz_path(self, rgb_path: Optional[str], depth_path: Optional[str]) -> Optional[str]:
        if depth_path:
            dp = Path(depth_path)
            if dp.suffix.lower() in (".tif", ".tiff"):
                return str(dp)
        if not rgb_path:
            return None
        rp = Path(rgb_path)
        parts = list(rp.parts)
        if "rgb" not in parts:
            return None
        idx = parts.index("rgb")
        parts[idx] = "xyz"
        xyz_dir = Path(*parts[:-1])
        for ext in (".tif", ".tiff"):
            cand = xyz_dir / (rp.stem + ext)
            if cand.exists():
                return str(cand)
        cands = glob.glob(str(xyz_dir / f"{rp.stem}.ti*"))
        return cands[0] if cands else None

    # ---------------- tensor normalization helpers ----------------
    def _rgb_denorm_to_01(self, x: torch.Tensor) -> torch.Tensor:
        if self.rgb_input_norm == "imagenet":
            mean = self.rgb_mean.to(x.device, x.dtype)
            std = self.rgb_std.to(x.device, x.dtype)
            y = x * std + mean
            return y.clamp(0, 1)
        if self.rgb_input_norm == "none":
            return x.clamp(0, 1)
        raise ValueError(f"Unsupported rgb_input_norm: {self.rgb_input_norm}")

    def _rgb_norm_from_01(self, x01: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        x01 = x01.to(device=like.device, dtype=like.dtype)
        if self.rgb_input_norm == "imagenet":
            mean = self.rgb_mean.to(like.device, like.dtype)
            std = self.rgb_std.to(like.device, like.dtype)
            return (x01 - mean) / std
        if self.rgb_input_norm == "none":
            return x01
        raise ValueError(f"Unsupported rgb_input_norm: {self.rgb_input_norm}")

    def _depth_denorm_to_01(self, d: torch.Tensor) -> torch.Tensor:
        mode = self.depth_input_norm
        d = d.float()
        if mode == "none":
            return d.clamp(0, 1)
        if mode == "m11":
            return ((d + 1.0) * 0.5).clamp(0, 1)
        if mode == "zscore":
            return (d * self.depth_std + self.depth_mean).clamp(0, 1)
        if mode == "auto":
            # auto detect common cases
            mn = float(torch.nan_to_num(d, nan=0.0).min())
            mx = float(torch.nan_to_num(d, nan=0.0).max())
            if mn >= -0.1 and mx <= 1.1:
                return d.clamp(0, 1)
            if mn >= -1.5 and mx <= 1.5:
                return ((d + 1.0) * 0.5).clamp(0, 1)
            return d.clamp(0, 1)
        raise ValueError(f"Unsupported depth_input_norm: {mode}")

    def _depth_norm_from_01(self, d01: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        d01 = d01.to(device=like.device, dtype=like.dtype).clamp(0, 1)
        mode = self.depth_input_norm
        if mode in ("none", "auto"):
            # "auto" cannot be perfectly inverted; assume [0,1] training depth unless overridden
            return d01
        if mode == "m11":
            return d01 * 2.0 - 1.0
        if mode == "zscore":
            return (d01 - self.depth_mean) / max(self.depth_std, 1e-8)
        raise ValueError(f"Unsupported depth_input_norm: {mode}")

    def _ensure_single_depth_channel(self, d: torch.Tensor) -> torch.Tensor:
        # accepts [B,1,H,W] or [B,3,H,W]
        if d.ndim != 4:
            raise ValueError(f"Expected depths [B,C,H,W], got {d.shape}")
        if d.shape[1] == 1:
            return d
        if d.shape[1] >= 1:
            warnings.warn(f"[LDM3DAnomalyGenerator] Depth has {d.shape[1]} channels; using channel 0 for LDM3D (expects RGB+1 depth).")
            return d[:, :1]
        raise ValueError(f"Invalid depth tensor shape: {d.shape}")

    # ---------------- mask sampling ----------------
    def _sample_mask_and_defect(
        self,
        cls: str,
        H: int,
        W: int,
        obj_mask01: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[str]]:
        files = self.mask_pool.get(cls, [])
        if not files:
            return None, None, None

        for _ in range(self.attempts_per_mask):
            src = random.choice(files)
            try:
                base = load_binary_mask(src)
            except Exception:
                continue
            if base.sum() == 0:
                # especially for EyeCandies all-black masks
                continue

            defect = self._defect_from_mask_path(src)
            if defect == "good":
                continue

            scale = random.uniform(self.scale_min, self.scale_max)
            angle = random.uniform(-self.rotate_max, self.rotate_max)
            hflip = random.random() < 0.5
            vflip = random.random() < 0.2

            aug = random_affine_mask(base, angle, scale, hflip, vflip)
            if self.use_elastic:
                aug = elastic_deform_mask(aug, alpha=self.alpha, sigma=self.sigma)

            placed = place_mask_on_canvas(aug, H, W)
            if placed.sum() == 0:
                continue

            if obj_mask01 is not None:
                inter = (placed & obj_mask01).astype(np.uint8)
                cov = float(inter.sum()) / float(max(1, placed.sum()))
                if cov < self.min_coverage:
                    continue
                final = inter
            else:
                final = placed

            if final.sum() == 0:
                continue

            return torch.from_numpy(final.astype(np.float32))[None], defect, src

        return None, None, None

    # ---------------- cache ----------------
    def _try_read_cache_rgbd(
        self,
        cls: str,
        sample_id: str,
        defect: str,
        H: int,
        W: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.cache_root:
            return None
        d = _cache_dir(self.cache_root, self.dataset, cls, sample_id, defect)
        idxs = _list_cached_anomalies(d)
        if len(idxs) < self.cache_k or len(idxs) == 0:
            return None

        pick = random.choice(idxs)
        rgb_p = os.path.join(d, f"anomaly_rgb_{pick:06d}.png")
        depth_p_npy = os.path.join(d, f"anomaly_depth_{pick:06d}.npy")
        depth_p_png = os.path.join(d, f"anomaly_depth_{pick:06d}.png")

        try:
            rgb = _pil_to_tensor01_rgb(Image.open(rgb_p).convert("RGB"))
            if rgb.shape[-2:] != (H, W):
                rgb = F.interpolate(rgb[None], size=(H, W), mode="bilinear", align_corners=False)[0]

            if os.path.isfile(depth_p_npy):
                depth_arr = np.load(depth_p_npy)
                depth = torch.from_numpy(depth_arr.astype(np.float32))
                if depth.ndim == 2:
                    depth = depth[None]
            elif os.path.isfile(depth_p_png):
                d16 = np.array(Image.open(depth_p_png))
                if d16.ndim == 3:
                    d16 = d16[..., 0]
                depth = _u16_to_tensor01_depth(d16.astype(np.uint16))
            else:
                return None

            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(depth[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            depth = depth.clamp(0, 1)
            return rgb.float(), depth.float()
        except Exception:
            return None

    def _write_cache_rgbd(
        self,
        cls: str,
        sample_id: str,
        defect: str,
        img_rgb01: torch.Tensor,
        img_depth01: torch.Tensor,
        mask01: torch.Tensor,
        anom_rgb01: torch.Tensor,
        anom_depth01: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> None:
        if not self.cache_root:
            return
        d = _cache_dir(self.cache_root, self.dataset, cls, sample_id, defect)
        _safe_mkdir(d)

        if self.save_original_once:
            op_rgb = os.path.join(d, "original_rgb.png")
            if not os.path.exists(op_rgb):
                _atomic_save_pil(_tensor01_to_pil_rgb(img_rgb01), op_rgb)

            op_depth = os.path.join(d, "original_depth.npy")
            if not os.path.exists(op_depth):
                _atomic_write_npy(img_depth01.detach().float().cpu().numpy(), op_depth)

        idx = _next_cache_index(d)
        _atomic_save_pil(_mask01_to_pil(mask01), os.path.join(d, f"mask_{idx:06d}.png"))
        _atomic_save_pil(_tensor01_to_pil_rgb(anom_rgb01), os.path.join(d, f"anomaly_rgb_{idx:06d}.png"))

        # depth cache: store .npy (precise) + optional png preview
        anom_depth_np = anom_depth01.detach().float().cpu().numpy()
        _atomic_write_npy(anom_depth_np, os.path.join(d, f"anomaly_depth_{idx:06d}.npy"))
        try:
            d16 = _tensor01_depth_to_u16(anom_depth01)
            _atomic_save_pil(Image.fromarray(d16, mode="I;16"), os.path.join(d, f"anomaly_depth_{idx:06d}.png"))
        except Exception:
            pass

        if self.save_meta and meta is not None:
            _atomic_write_json(meta, os.path.join(d, f"meta_{idx:06d}.json"))

    # ---------------- LDM3D SDEdit core ----------------
    def _mask_to_tensor(self, mask: np.ndarray, H: int, W: int) -> torch.Tensor:
        m = Image.fromarray((mask.astype(np.uint8) * 255), "L")
        if self.resolution is not None:
            tgt_h = tgt_w = self.resolution
        else:
            tgt_h, tgt_w = H, W
        if (m.size[1], m.size[0]) != (tgt_h, tgt_w):
            m = m.resize((tgt_w, tgt_h), Image.NEAREST)
        arr = (np.array(m) > 127).astype(np.float32)
        return torch.from_numpy(arr)[None]  # [1,H,W]

    @torch.no_grad()
    def _sdeedit_single_rgbd(
        self,
        rgb01: torch.Tensor,    # [3,H,W] in [0,1]
        depth01: torch.Tensor,  # [1,H,W] in [0,1]
        mask01: torch.Tensor,   # [1,H,W] in {0,1} at source res
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns anomaly RGB/depth in [0,1] at original input size.
        """
        H, W = rgb01.shape[-2], rgb01.shape[-1]
        if depth01.shape[-2:] != (H, W):
            depth01 = F.interpolate(depth01[None], size=(H, W), mode="bilinear", align_corners=False)[0]

        tgt_h = self.resolution if self.resolution is not None else H
        tgt_w = self.resolution if self.resolution is not None else W

        # resize to model working size if requested
        if (tgt_h, tgt_w) != (H, W):
            rgb01_rs = F.interpolate(rgb01[None], size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)[0]
            depth01_rs = F.interpolate(depth01[None], size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)[0]
            mask01_rs = F.interpolate(mask01[None], size=(tgt_h, tgt_w), mode="nearest")[0]
        else:
            rgb01_rs, depth01_rs, mask01_rs = rgb01, depth01, mask01

        rgb01_rs = rgb01_rs.clamp(0, 1)
        depth01_rs = depth01_rs.clamp(0, 1)
        mask01_rs = (mask01_rs > 0.5).float()

        # LDM3D expects 4ch normalized to [-1,1]
        x0_rgbd = torch.cat([rgb01_rs, depth01_rs], dim=0)[None].to(self.device, dtype=self.pipe.vae.dtype)
        x0_rgbd = x0_rgbd * 2.0 - 1.0

        z0 = self.pipe.vae.encode(x0_rgbd).latent_dist.sample() * self.pipe.vae.config.scaling_factor

        # latent mask
        mask_t = mask01_rs[None].to(self.device, dtype=torch.float32)  # [1,1,H,W]
        mask_lat = F.interpolate(mask_t, size=z0.shape[-2:], mode="nearest").to(dtype=z0.dtype)

        # scheduler setup
        self.pipe.scheduler.set_timesteps(self.steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        t0_index = int(min(self.steps * self.strength, self.steps - 1))
        t_start = max(self.steps - t0_index - 1, 0)
        timesteps = timesteps[t_start:]

        # always do cfg because we have cond/uncond embeddings prepared
        cond_emb = self._cond_emb
        uncond_emb = self._uncond_emb
        do_cfg = True

        # noise init for SDEdit
        z0_f = z0.float()
        eps0 = torch.randn_like(z0_f)
        t = timesteps[0]
        t_batch = torch.tensor([int(t.item())], device=self.device, dtype=torch.long)
        zt = self.pipe.scheduler.add_noise(z0_f, eps0, t_batch)

        for step_idx, t in enumerate(timesteps):
            latent_model_input = zt
            if do_cfg:
                latent_model_input = torch.cat([zt, zt], dim=0)

            if hasattr(self.pipe.scheduler, "scale_model_input"):
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            emb = torch.cat([uncond_emb, cond_emb], dim=0)
            latent_model_input_unet = latent_model_input.to(dtype=self.pipe.unet.dtype)

            noise_pred = self.pipe.unet(
                latent_model_input_unet,
                t,
                encoder_hidden_states=emb,
            ).sample.float()

            eps_u, eps_c = noise_pred.chunk(2, dim=0)
            eps = eps_u + self.guidance * (eps_c - eps_u)

            zt = self.pipe.scheduler.step(eps, t, zt).prev_sample.float()
            zt = torch.nan_to_num(zt, nan=0.0, posinf=0.0, neginf=0.0)

            # clamp outside ROI to reference noised latent at next timestep
            if step_idx < len(timesteps) - 1:
                t_next = timesteps[step_idx + 1]
                t_next_batch = torch.tensor([int(t_next.item())], device=self.device, dtype=torch.long)
                z_ref_next = self.pipe.scheduler.add_noise(z0_f, eps0, t_next_batch)
            else:
                z_ref_next = z0_f
            zt = mask_lat.float() * zt + (1.0 - mask_lat.float()) * z_ref_next

        x_rec = self.pipe.vae.decode(zt.to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor).sample.float()
        x_rec = x_rec.clamp(-1, 1)
        x0 = x0_rgbd.float()

        # pixel-space composite to preserve background strictly
        mask_px = mask_t.to(dtype=x_rec.dtype)
        x_mix = mask_px * x_rec + (1.0 - mask_px) * x0

        x_mix01 = (x_mix + 1.0) * 0.5
        x_mix01 = x_mix01.clamp(0, 1)

        out_rgb = x_mix01[0, :3]
        out_depth = x_mix01[0, 3:4]

        if (tgt_h, tgt_w) != (H, W):
            out_rgb = F.interpolate(out_rgb[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            out_depth = F.interpolate(out_depth[None], size=(H, W), mode="bilinear", align_corners=False)[0]

        return out_rgb.clamp(0, 1), out_depth.clamp(0, 1)

    # ---------------- public call ----------------
    @torch.no_grad()
    def __call__(
        self,
        imgs: torch.Tensor,
        labels: Optional[Sequence[Any]] = None,
        depths: Optional[torch.Tensor] = None,
        paths: Optional[Sequence[str]] = None,
        depth_paths: Optional[Sequence[str]] = None,
    ):
        """
        Inputs:
            imgs   : [B,3,H,W] in training normalization (usually ImageNet)
            depths : [B,1,H,W] or [B,C,H,W] in training depth normalization
            paths/depth_paths are used for class/sample-id inference + MVTec3D obj-mask.
        Returns:
            (imgs, (imgs_abn, depths_abn))  # both anomaly outputs are in SAME normalization space as inputs
        """
        if depths is None:
            if self.on_failure == "raise":
                raise ValueError("LDM3DSDEditAnomalyGenerator requires depths for RGBD synthesis.")
            return imgs, (imgs.clone(), None)

        B, _, H, W = imgs.shape
        depths = self._ensure_single_depth_channel(depths)

        imgs_rgb01 = self._rgb_denorm_to_01(imgs.detach())
        depths01 = self._depth_denorm_to_01(depths.detach())

        out_rgb01 = imgs_rgb01.clone()
        out_depth01 = depths01.clone()

        if paths is None:
            paths = [f"sample_{i:06d}.png" for i in range(B)]
        if depth_paths is None:
            depth_paths = [None] * B

        for i in range(B):
            rgb_path = Path(str(paths[i]))
            cls = self._infer_cls_from_path(rgb_path)
            sample_id = self._infer_sample_id_from_path(rgb_path)

            # optional MVTec3D object mask from xyz
            obj_mask = None
            if self.dataset == "mvtec3d" and self.use_obj_intersection_if_possible:
                xyz_path = self._infer_xyz_path(str(paths[i]) if paths is not None else None,
                                               str(depth_paths[i]) if depth_paths is not None and depth_paths[i] is not None else None)
                if xyz_path and os.path.isfile(xyz_path):
                    try:
                        obj_mask = compute_object_mask_from_xyz(xyz_path)
                        if obj_mask.shape != (H, W):
                            obj_mask = (np.array(Image.fromarray(obj_mask * 255).resize((W, H), Image.NEAREST)) > 0).astype(np.uint8)
                        if obj_mask.sum() < 50:
                            obj_mask = None
                    except Exception:
                        obj_mask = None

            mask_t, defect, src_mask_path = self._sample_mask_and_defect(cls, H, W, obj_mask01=obj_mask)
            if mask_t is None or defect is None:
                continue

            cached = self._try_read_cache_rgbd(cls, sample_id, defect, H, W)
            if cached is not None:
                crgb, cdepth = cached
                out_rgb01[i] = crgb.to(out_rgb01.device, out_rgb01.dtype)
                out_depth01[i] = cdepth.to(out_depth01.device, out_depth01.dtype)
                continue

            try:
                an_rgb, an_depth = self._sdeedit_single_rgbd(
                    rgb01=imgs_rgb01[i].float(),
                    depth01=depths01[i].float(),
                    mask01=mask_t.float(),
                )
                out_rgb01[i] = an_rgb.to(out_rgb01.device, out_rgb01.dtype)
                out_depth01[i] = an_depth.to(out_depth01.device, out_depth01.dtype)

                meta = {
                    "dataset": self.dataset,
                    "class": cls,
                    "sample_id": sample_id,
                    "defect_type": defect,
                    "prompt": self.defect_prompt,            # generic prompt
                    "negative_prompt": self.clean_prompt if self.use_clean_as_negative else "",
                    "steps": self.steps,
                    "strength": self.strength,
                    "guidance": self.guidance,
                    "src_mask_path": src_mask_path,
                    "cache_policy": {"cache_k": self.cache_k},
                }
                self._write_cache_rgbd(
                    cls=cls,
                    sample_id=sample_id,
                    defect=defect,
                    img_rgb01=imgs_rgb01[i],
                    img_depth01=depths01[i],
                    mask01=mask_t,
                    anom_rgb01=an_rgb,
                    anom_depth01=an_depth,
                    meta=meta,
                )
            except Exception as e:
                if self.on_failure == "raise":
                    raise
                warnings.warn(f"[LDM3DAnomalyGenerator] failed on {paths[i]}: {e}")
                continue

        imgs_abn = self._rgb_norm_from_01(out_rgb01, like=imgs)
        depths_abn = self._depth_norm_from_01(out_depth01, like=depths)

        # preserve exact dtype/device
        imgs_abn = imgs_abn.to(device=imgs.device, dtype=imgs.dtype)
        depths_abn = depths_abn.to(device=depths.device, dtype=depths.dtype)

        return imgs, (imgs_abn, depths_abn)
