from __future__ import annotations
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra


def load_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa
        except ModuleNotFoundError:
            sys.exit("PyYAML is required for YAML configs. Install via `pip install pyyaml`.")
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    else:
        return json.loads(path.read_text())


# ---------------------------
# Original sampler (2D datasets)
# ---------------------------
def sample_images(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    train_subpaths: Tuple[str, ...],
    allowed_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    rename_images: bool = True,
) -> None:
    allowed_exts = tuple(ext.lower() for ext in allowed_exts)

    target_train_root = target_root / "train"
    target_train_root.mkdir(parents=True, exist_ok=True)

    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name

        imgs: List[Path] = []
        chosen_subpath = None
        for sub in train_subpaths:
            candidate = category_dir / sub
            if candidate.is_dir():
                found = [
                    p for p in candidate.iterdir()
                    if p.is_file() and p.suffix.lower() in allowed_exts
                ]
                if found:
                    imgs = found
                    chosen_subpath = sub
                    break

        if not imgs:
            print(f"[skip] {cat_name}: none of {list(train_subpaths)} contained images")
            continue

        k = min(num_samples, len(imgs))
        random.shuffle(imgs)
        selected = imgs[:k]

        dest_dir = target_train_root / cat_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        existing_files = [
            p for p in dest_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        start_idx = len(existing_files)

        for i, src in enumerate(selected):
            new_name = (f"{start_idx + i:03d}{src.suffix.lower()}") if rename_images else src.name
            shutil.copy2(src, dest_dir / new_name)

        print(f"[✓] {cat_name}: copied {k}/{len(imgs)} from '{chosen_subpath}'")


# ---------------------------
# Helpers for RGBD sampling
# ---------------------------
def _iter_files_with_ext(root: Path, allowed_exts: Tuple[str, ...]) -> List[Path]:
    """Return files directly under root with allowed extensions."""
    if not root.is_dir():
        return []
    exts = tuple(e.lower() for e in allowed_exts)
    return [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]


def _match_by_stem(rgb_files: List[Path], depth_files: List[Path]) -> List[Tuple[Path, Path]]:
    """Match two file lists by Path.stem."""
    depth_map = {p.stem: p for p in depth_files}
    pairs: List[Tuple[Path, Path]] = []
    for r in rgb_files:
        d = depth_map.get(r.stem)
        if d is not None:
            pairs.append((r, d))
    return pairs


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _existing_count(dest_dir: Path, allowed_exts: Tuple[str, ...]) -> int:
    if not dest_dir.is_dir():
        return 0
    exts = tuple(e.lower() for e in allowed_exts)
    return sum(1 for p in dest_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


# ---------------------------
# MVTec 3D-AD sampler (RGB + XYZ)
# ---------------------------
def sample_rgbd_mvtec3d(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    rgb_train_subpath: str = "train/good/rgb",
    depth_train_subpath: str = "train/good/xyz",
    allowed_rgb_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    allowed_depth_exts: Tuple[str, ...] = (".tiff", ".tif", ".png", ".exr"),
    rename_images: bool = True,
    target_rgb_dirname: str = "rgb",
    target_depth_dirname: str = "xyz",
    copy_calibration: bool = True,
) -> None:
    """Few-shot sampler for MVTec 3D-AD.

    Expected per-class structure:
      <class>/train/good/rgb/*
      <class>/train/good/xyz/*
    We sample matched (rgb, xyz) pairs by identical stem.
    """
    target_train_root = target_root / "train"
    _ensure_dir(target_train_root)

    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name

        rgb_dir = category_dir / rgb_train_subpath
        depth_dir = category_dir / depth_train_subpath
        rgb_files = _iter_files_with_ext(rgb_dir, allowed_rgb_exts)
        depth_files = _iter_files_with_ext(depth_dir, allowed_depth_exts)
        pairs = _match_by_stem(rgb_files, depth_files)

        if not pairs:
            print(f"[skip] {cat_name}: no matched rgb/xyz pairs under '{rgb_train_subpath}' & '{depth_train_subpath}'")
            continue

        random.shuffle(pairs)
        selected = pairs[: min(num_samples, len(pairs))]

        dest_cat = target_train_root / cat_name
        dest_rgb = dest_cat / target_rgb_dirname
        dest_depth = dest_cat / target_depth_dirname
        _ensure_dir(dest_rgb)
        _ensure_dir(dest_depth)

        start_idx = _existing_count(dest_rgb, allowed_rgb_exts)

        for i, (rgb_src, depth_src) in enumerate(selected):
            if rename_images:
                stem = f"{start_idx + i:03d}"
                rgb_name = stem + rgb_src.suffix.lower()
                depth_name = stem + depth_src.suffix.lower()
            else:
                rgb_name = rgb_src.name
                depth_name = depth_src.name

            shutil.copy2(rgb_src, dest_rgb / rgb_name)
            shutil.copy2(depth_src, dest_depth / depth_name)

        if copy_calibration:
            calib = category_dir / "calibration"
            if calib.is_dir():
                dest_calib = dest_cat / "calibration"
                if not dest_calib.exists():
                    shutil.copytree(calib, dest_calib)

        print(f"[✓] {cat_name}: copied {len(selected)}/{len(pairs)} rgb+xyz pairs")


# ---------------------------
# EyeCandies sampler (RGB + depth)
# ---------------------------
def _eyecandies_parse_prefix(stem: str) -> Optional[str]:
    # e.g. "000_image_0" -> "000"
    if "_image_" in stem:
        return stem.split("_image_", 1)[0]
    return None


def sample_rgbd_eyecandies(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    rgb_glob: str = "train/data/*_image_*.png",
    depth_rel_template: str = "train/data/{prefix}_depth.png",
    allowed_depth_exts: Tuple[str, ...] = (".png", ".tiff", ".tif", ".exr"),
    rename_images: bool = True,
    target_rgb_dirname: str = "rgb",
    target_depth_dirname: str = "depth",
) -> None:
    """Few-shot sampler for EyeCandies.

    Expected per-object structure:
      <object>/train/data/<id>_image_*.png
      <object>/train/data/<id>_depth.png
    We pair each *_image_* with its corresponding <id>_depth.*
    """
    target_train_root = target_root / "train"
    _ensure_dir(target_train_root)

    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name
        rgb_files = sorted(category_dir.glob(rgb_glob))
        rgb_files = [p for p in rgb_files if p.is_file()]

        pairs: List[Tuple[Path, Path]] = []
        allowed_depth_exts_l = tuple(e.lower() for e in allowed_depth_exts)

        for rgb in rgb_files:
            prefix = _eyecandies_parse_prefix(rgb.stem)
            if not prefix:
                continue
            depth = category_dir / depth_rel_template.format(prefix=prefix)
            if depth.is_file() and depth.suffix.lower() in allowed_depth_exts_l:
                pairs.append((rgb, depth))

        if not pairs:
            print(f"[skip] {cat_name}: no matched rgb/depth pairs with rgb_glob='{rgb_glob}'")
            continue

        random.shuffle(pairs)
        selected = pairs[: min(num_samples, len(pairs))]

        dest_cat = target_train_root / cat_name
        dest_rgb = dest_cat / target_rgb_dirname
        dest_depth = dest_cat / target_depth_dirname
        _ensure_dir(dest_rgb)
        _ensure_dir(dest_depth)

        start_idx = _existing_count(dest_rgb, (".png", ".jpg", ".jpeg"))

        for i, (rgb_src, depth_src) in enumerate(selected):
            if rename_images:
                stem = f"{start_idx + i:03d}"
                rgb_name = stem + rgb_src.suffix.lower()
                depth_name = stem + depth_src.suffix.lower()
            else:
                rgb_name = rgb_src.name
                depth_name = depth_src.name

            shutil.copy2(rgb_src, dest_rgb / rgb_name)
            shutil.copy2(depth_src, dest_depth / depth_name)

        print(f"[✓] {cat_name}: copied {len(selected)}/{len(pairs)} rgb+depth pairs")


# ---------------------------
# Hydra config
# ---------------------------
@dataclass
class SamplerCfg:
    source: str = ""
    target: str = ""

    # "generic" | "mvtec3d" | "eyecandies"
    dataset: str = "generic"

    num_samples: int = 1

    # -------- generic (MVTec AD / ViSA style) --------
    train_subpaths: List[str] = field(default_factory=lambda: ["train/good", "train/ok"])
    allowed_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    rename_images: bool = True

    # -------- mvtec3d --------
    rgb_train_subpath: str = "train/good/rgb"
    depth_train_subpath: str = "train/good/xyz"
    allowed_rgb_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    allowed_depth_exts: List[str] = field(default_factory=lambda: [".tiff", ".tif", ".png", ".exr"])
    target_rgb_dirname: str = "rgb"
    target_depth_dirname: str = "xyz"
    copy_calibration: bool = True

    # -------- eyecandies --------
    rgb_glob: str = "train/data/*_image_*.png"
    depth_rel_template: str = "train/data/{prefix}_depth.png"
    eyecandies_target_depth_dirname: str = "depth"

    seed: int | None = None
    user_config: str = ""


def _finalize_cfg(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.user_config:
        ext = load_config_file(Path(cfg.user_config))
        cfg = OmegaConf.merge(OmegaConf.structured(SamplerCfg), cfg, ext)

    source = cfg.get("source")
    target = cfg.get("target")
    if not source or not target:
        sys.exit("Both 'source' and 'target' must be specified. "
                 "Provide via CLI (source=..., target=...) or user_config=...")

    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_few_shot")
def main(cfg: DictConfig) -> None:
    cfgd = _finalize_cfg(cfg)
    if cfgd.get("seed") is not None:
        random.seed(int(cfgd["seed"]))

    dataset = str(cfgd.get("dataset", "generic")).lower()

    if dataset in {"mvtec3d", "mvtec_3d", "mvtec3d_ad", "mvtec-3d"}:
        sample_rgbd_mvtec3d(
            source_root=Path(cfgd["source"]),
            target_root=Path(cfgd["target"]),
            num_samples=int(cfgd["num_samples"]),
            rgb_train_subpath=str(cfgd.get("rgb_train_subpath", "train/good/rgb")),
            depth_train_subpath=str(cfgd.get("depth_train_subpath", "train/good/xyz")),
            allowed_rgb_exts=tuple(cfgd.get("allowed_rgb_exts", cfgd.get("allowed_exts"))),
            allowed_depth_exts=tuple(cfgd.get("allowed_depth_exts", [".tiff", ".tif", ".png", ".exr"])),
            rename_images=bool(cfgd.get("rename_images", True)),
            target_rgb_dirname=str(cfgd.get("target_rgb_dirname", "rgb")),
            target_depth_dirname=str(cfgd.get("target_depth_dirname", "xyz")),
            copy_calibration=bool(cfgd.get("copy_calibration", True)),
        )

    elif dataset in {"eyecandies", "eye_candies", "eye-candies"}:
        sample_rgbd_eyecandies(
            source_root=Path(cfgd["source"]),
            target_root=Path(cfgd["target"]),
            num_samples=int(cfgd["num_samples"]),
            rgb_glob=str(cfgd.get("rgb_glob", "train/data/*_image_*.png")),
            depth_rel_template=str(cfgd.get("depth_rel_template", "train/data/{prefix}_depth.png")),
            allowed_depth_exts=tuple(cfgd.get("allowed_depth_exts", [".png", ".tiff", ".tif", ".exr"])),
            rename_images=bool(cfgd.get("rename_images", True)),
            target_rgb_dirname=str(cfgd.get("target_rgb_dirname", "rgb")),
            target_depth_dirname=str(cfgd.get("eyecandies_target_depth_dirname", "depth")),
        )

    else:
        sample_images(
            source_root=Path(cfgd["source"]),
            target_root=Path(cfgd["target"]),
            num_samples=int(cfgd["num_samples"]),
            train_subpaths=tuple(cfgd["train_subpaths"]),
            allowed_exts=tuple(cfgd["allowed_exts"]),
            rename_images=bool(cfgd["rename_images"]),
        )


if __name__ == "__main__":
    main()
