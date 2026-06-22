import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image

from src.datasets.dataset import build_dataloader, _read_depth_any
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule
from src.utils.depth_repr_utils import raw_depth_batch_to_model_repr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any], data_cfg: Dict[str, Any]) -> VisionModule:
    """Build dual-predictor VisionModule.

    RGB-only checkpoints remain supported when use_depth=False.
    For RGB-D, RGB and depth are reconstructed separately and fused only at score level.
    """
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        use_depth=bool(data_cfg.get("use_depth", False)),
    )


def _load_dual_predictor_ckpt(model: VisionModule, state: Dict[str, Any], use_depth: bool) -> None:
    if "predictor_rgb" in state:
        model.predictor_rgb.load_state_dict(state["predictor_rgb"])
    elif "predictor" in state:
        model.predictor_rgb.load_state_dict(state["predictor"])
    else:
        raise KeyError("Checkpoint does not contain predictor_rgb (or legacy predictor).")

    if use_depth and model.predictor_d is not None:
        if "predictor_d" not in state or state["predictor_d"] is None:
            raise KeyError("Depth is enabled but checkpoint does not contain predictor_d.")
        model.predictor_d.load_state_dict(state["predictor_d"])

    if model.projector is not None and state.get("projector", None) is not None:
        model.projector.load_state_dict(state["projector"])


def _fuse_branch_scores(score_rgb: torch.Tensor, score_d: torch.Tensor | None, mode: str = "max") -> torch.Tensor:
    if score_d is None:
        return score_rgb
    m = str(mode).lower()
    if m == "max":
        return torch.maximum(score_rgb, score_d)
    if m in ("avg", "mean"):
        return 0.5 * (score_rgb + score_d)
    if m in ("sum", "add"):
        return score_rgb + score_d
    if m == "rgb":
        return score_rgb
    if m in ("depth", "d"):
        return score_d
    raise ValueError(f"Unknown score_fusion={mode}. Expected max/avg/sum/rgb/depth.")


def _compute_patch_error(
    model: VisionModule,
    img: torch.Tensor,
    paths: List[str],
    n_layer: int,
    depths: torch.Tensor | None = None,
    depth_paths: List[str] | None = None,
    score_fusion: str = "max",
) -> torch.Tensor:
    feats = model.target_features(img, paths, n_layer=n_layer, depths=depths, depth_paths=depth_paths)
    pred_rgb = model.predict(feats["rgb"], modality="rgb")
    err_rgb = F.mse_loss(feats["rgb"], pred_rgb, reduction="none").mean(dim=2)

    err_d = None
    if "depth" in feats:
        pred_d = model.predict(feats["depth"], modality="depth")
        err_d = F.mse_loss(feats["depth"], pred_d, reduction="none").mean(dim=2)

    return _fuse_branch_scores(err_rgb, err_d, mode=score_fusion)


def _resolve_raw_as_single_depth(cfg: Dict[str, Any]) -> bool:
    data_cfg = cfg.get("data", {})
    if "raw_as_single_depth" in data_cfg:
        return bool(data_cfg["raw_as_single_depth"])
    depth_repr = str(data_cfg.get("depth_repr", "raw")).lower()
    synth_type = str(cfg.get("anomaly_synth", {}).get("type", "")).lower()
    return bool(depth_repr == "raw" and synth_type == "ldm3d")


def _prepare_depths_for_model(depths: torch.Tensor | None, cfg: Dict[str, Any], device: torch.device):
    if depths is None:
        return None
    return raw_depth_batch_to_model_repr(
        depths.to(device, non_blocking=True),
        depth_repr=str(cfg["data"].get("depth_repr", "raw")).lower(),
        raw_as_single_depth=_resolve_raw_as_single_depth(cfg),
    )


def _find_demo_depth_path(image_path: Path, dataset_name: str, cfg: Dict[str, Any]) -> Path | None:
    dataset_name = dataset_name.lower()

    if dataset_name == "eyecandies":
        stem = image_path.stem
        # Stem pattern "<prefix>_image_<view>"
        if "_image_" in stem:
            prefix = stem.split("_image_", 1)[0]
        else:
            prefix = stem

        # Pattern 1: Standard structure - depth file in the same directory
        # e.g. .../data/0_image_0.png -> .../data/0_depth.png
        dpath = image_path.parent / f"{prefix}_depth.png"
        if dpath.is_file():
            return dpath

        # Pattern 2: Flat assets structure - rgb in <class>/rgb/, depth in <class>/depth/
        # e.g. .../CandyCane/rgb/38_image_0.png -> .../CandyCane/depth/38_depth.png
        if image_path.parent.name == "rgb":
            depth_dir = image_path.parent.parent / "depth"
            cand = depth_dir / f"{prefix}_depth.png"
            if cand.is_file():
                return cand
            # Try other extensions
            for ext in (".png", ".tiff", ".tif", ".exr", ".npy"):
                cand = depth_dir / f"{prefix}_depth{ext}"
                if cand.is_file():
                    return cand
            # Try just <prefix>.png (in case the naming convention is different)
            for ext in (".png", ".tiff", ".tif", ".exr", ".npy"):
                cand = depth_dir / f"{prefix}{ext}"
                if cand.is_file():
                    return cand

        return None

    if dataset_name == "mvtec3d":
        rgb_dir = image_path.parent
        rgb_dirname = cfg["data"].get("mvtec3d_rgb_dirname", "rgb")
        if rgb_dir.name != rgb_dirname:
            logger.debug("Expected rgb dir '%s' but got '%s' for %s",
                         rgb_dirname, rgb_dir.name, image_path)
        depth_dir = rgb_dir.parent / cfg["data"].get("mvtec3d_depth_dirname", "xyz")
        stem = image_path.stem
        allowed_exts = tuple(
            str(x).lower()
            for x in cfg["data"].get("allowed_depth_exts", [".tiff", ".tif", ".png", ".exr"])
        )
        for ext in allowed_exts:
            cand = depth_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        return None

    return None



@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"], cfg.get("data", {}))
    state = torch.load(ckpt, map_location="cpu")
    _load_dual_predictor_ckpt(model, state, use_depth=bool(cfg["data"].get("use_depth", False)))
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    # error = cfg["meta"].get("loss_mode", "l2")

    dataset_name = cfg["data"].get("dataset", "mvtec").lower()
    if dataset_name == "mvtec":
        classnames = cfg["data"]["mvtec_classnames"]
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == "visa":
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    elif dataset_name == "mvtec3d":
        if "mvtec3d_classnames" not in cfg["data"]:
            raise KeyError("cfg.data.mvtec3d_classnames is required for dataset=mvtec3d")
        classnames = cfg["data"]["mvtec3d_classnames"]
        K = cfg["testing"].get("K_top_mvtec3d", cfg["testing"].get("K_top_mvtec"))
    elif dataset_name == "eyecandies":
        if "eyecandies_classnames" not in cfg["data"]:
            raise KeyError("cfg.data.eyecandies_classnames is required for dataset=eyecandies")
        classnames = cfg["data"]["eyecandies_classnames"]
        K = cfg["testing"].get("K_top_eyecandies", cfg["testing"].get("K_top_mvtec"))
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    test_root_str = str(cfg["data"]["test_root"])
    if dataset_name.lower() not in test_root_str.lower():
        logger.warning("dataset=%s but test_root=%s (name not found in path); continue anyway.", dataset_name, test_root_str)

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
            use_depth=bool(cfg["data"].get("use_depth", False)),
            depth_repr="raw",
            eyecandies_test_split=str(cfg["data"].get("eyecandies_test_split", "test_public")),
            eyecandies_view_idx=int(cfg["data"].get("eyecandies_view_idx", 0)),
            eyecandies_use_union_mask=bool(cfg["data"].get("eyecandies_use_union_mask", True)),
            eyecandies_return_parts_masks=bool(cfg["data"].get("eyecandies_return_parts_masks", False)),
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]
            name_buf.extend(batch["image_name"])
            is_anom = batch["is_anomaly"]
            if torch.is_tensor(is_anom):
                labels.extend(is_anom.detach().cpu().tolist())
            else:
                labels.extend(list(is_anom))

            depths = batch.get("depth", None)
            depth_paths = batch.get("depth_path", None)
            if depths is not None and bool(cfg["data"].get("use_depth", False)):
                depths_model = _prepare_depths_for_model(depths, cfg, device)
            else:
                depths_model = None
                depth_paths = None

            l = _compute_patch_error(
                model,
                img,
                paths,
                n_layer=n_layer,
                depths=depths_model,
                depth_paths=depth_paths,
                score_fusion=cfg["testing"].get("score_fusion", "max"),
            )

            topk = torch.topk(l, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.detach().cpu().tolist())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(img.cpu()); mask_buf.append(mask.cpu())

        p_np = np.asarray(patch_scores, dtype=np.float32)
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)
            # --- Save clean overlay images for paper ---
            overlay_dir = Path(cfg["logging"]["folder"]) / "overlay" / cls
            overlay_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(len(name_buf)):
                rgb_uint8 = (imgs_un[idx].clip(0, 1) * 255).astype(np.uint8)
                heat = pix_norm[idx]
                import cv2
                heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
                rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(heat_color, 0.5, rgb_bgr, 0.5, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                # 把路径中的/替换为_，并去掉已有的.png后缀再重新加
                safe_name = name_buf[idx].replace("/", "_")
                if safe_name.endswith(".png"):
                    safe_name = safe_name[:-4]
                Image.fromarray(overlay_rgb).save(overlay_dir / f"{safe_name}.png")



    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"], cfg.get("data", {}))
    state = torch.load(ckpt, map_location="cpu")
    _load_dual_predictor_ckpt(model, state, use_depth=bool(cfg["data"].get("use_depth", False)))
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec").lower()
    test_root_str = str(cfg["data"]["test_root"])
    if dataset_name not in test_root_str.lower():
        logger.warning(
            "dataset=%s but test_root=%s (name not found in path); continue anyway.",
            dataset_name, test_root_str,
        )

    test_root = Path(cfg["data"]["test_root"])

    # -------- Collect image paths (support both standard and flat structures) --------
    img_paths: List[Path] = []
    rgb_dirname = str(cfg["data"].get("mvtec3d_rgb_dirname", "rgb"))

    img_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    if dataset_name == "mvtec3d":
        # Pattern 1: standard MVTec 3D-AD structure: <class>/test/<defect>/rgb/*
        for ext in img_exts:
            img_paths += list(test_root.glob(f"*/test/*/{rgb_dirname}/{ext}"))

        # Pattern 2: flat assets structure: <class>/<defect>/rgb/*  (no 'test' dir)
        if not img_paths:
            for ext in img_exts:
                img_paths += list(test_root.glob(f"*/*/{rgb_dirname}/{ext}"))

        # Pattern 3: even flatter: <defect>/rgb/*  (single class)
        if not img_paths:
            for ext in img_exts:
                img_paths += list(test_root.glob(f"*/{rgb_dirname}/{ext}"))

    elif dataset_name == "eyecandies":
        split = str(cfg["data"].get("eyecandies_test_split", "test_public"))
        view = int(cfg["data"].get("eyecandies_view_idx", 0))
        # Pattern 1: standard structure <class>/<split>/data/*_image_<view>.png
        img_paths = sorted(test_root.glob(f"*/{split}/data/*_image_{view}.png"))
        # Pattern 2: flat assets structure <class>/rgb/*_image_<view>.png
        if not img_paths:
            img_paths = sorted(test_root.glob(f"*/rgb/*_image_{view}.png"))
        # Pattern 3: any image in <class>/rgb/ (view-agnostic fallback)
        if not img_paths:
            for ext in img_exts:
                img_paths += list(test_root.glob(f"*/rgb/{ext}"))


    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
        for ext in exts:
            img_paths += list(test_root.rglob(ext))

    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(
            f"No images found under: {test_root}\n"
            f"Tried patterns for dataset={dataset_name}. "
            f"Please check your directory structure."
        )
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")
        W0, H0 = pil.size
        pil_resized = pil.resize((crop, crop), Image.BILINEAR)
        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)
        img = (img - mean) / std
        return pil, (W0, H0), img

    def _to_numpy_image(t_img: torch.Tensor):
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()
        return (x * 255.0).astype(np.uint8)

    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        import cv2
        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    def _load_demo_depth_any(depth_path: Path) -> torch.Tensor:
        t = _read_depth_any(str(depth_path), crop).float()
        if t.ndim != 3:
            raise ValueError(f"Expected CHW depth tensor, got {tuple(t.shape)} for {depth_path}")
        return t.unsqueeze(0).to(device)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        # Find matching depth file (supports both standard and flat structures)
        depths = None
        depth_paths = None
        if bool(cfg["data"].get("use_depth", False)):
            dpath = _find_demo_depth_path(path, dataset_name, cfg)
            if dpath is not None:
                depths_raw = _load_demo_depth_any(dpath)
                depths = raw_depth_batch_to_model_repr(
                    depths_raw,
                    depth_repr=str(cfg["data"].get("depth_repr", "raw")).lower(),
                    raw_as_single_depth=_resolve_raw_as_single_depth(cfg),
                )
                depth_paths = [str(dpath)]
            else:
                logger.warning("No matching depth found for %s, skipping depth branch for this sample.", path)

        l = _compute_patch_error(
            model,
            img,
            [str(path)],
            n_layer=n_layer,
            depths=depths,
            depth_paths=depth_paths,
            score_fusion=cfg["testing"].get("score_fusion", "max"),
        )

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)
        pix = pix.squeeze(0).squeeze(0)

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)

        img_uint8 = _to_numpy_image(img)

        # Relative path for output
        try:
            rel = path.relative_to(test_root)
        except ValueError:
            rel = Path(path.name)
        save_dir = out_root / rel.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")



def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()