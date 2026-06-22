from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast

from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.utils.depth_repr_utils import (
    raw_depth_batch_to_model_repr,
    raw_depth_batch_to_single_channel_depth,
)
from src.utils.synthesis import CutPasteUnion
from src.foundad import VisionModule


_GLOBAL_SEED = 0
random.seed(42)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainableModules(nn.Module):
    def __init__(self, model: VisionModule):
        super().__init__()
        self.predictor_rgb = model.predictor_rgb
        if model.predictor_d is not None:
            self.predictor_d = model.predictor_d
        if model.projector is not None:
            self.projector = model.projector


class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        dcfg = args["data"]
        self.model = VisionModule(
            mcfg["model"],
            mcfg["pred_depth"],
            mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True),
            feat_normed=mcfg.get("feat_normed", False),
            use_depth=dcfg.get("use_depth", False),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.trainable_modules = TrainableModules(self.model)
        self.trainable_modules.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2")  # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")
        self.model_depth_repr = str(dcfg.get("depth_repr", "raw")).lower()
        self.raw_as_single_depth = bool(dcfg.get("raw_as_single_depth", False))

        # ---------- data ----------
        assert dcfg["dataset"] in dcfg["data_name"], "dcfg['dataset'] should appear in few-shot folder name"
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip", False),
            use_vflip=dcfg.get("use_vflip", False),
            use_rotate90=dcfg.get("use_rotate90", False),
            use_color_jitter=dcfg.get("use_color_jitter", False),
            use_gray=dcfg.get("use_gray", False),
            use_blur=dcfg.get("use_blur", False),
            datasetname=dcfg.get("dataset", None),
            use_depth=dcfg.get("use_depth", False),
            depth_repr="raw",
        )
        self.batch_size = dcfg["batch_size"]

        # ---------- anomaly synthesis (CutPaste / LDM3D) ----------
        self.anomaly_gen_kind, self.anomaly_gen = self._build_anomaly_generator(args)
        logger.info("Anomaly generator = %s", self.anomaly_gen_kind)

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.trainable_modules,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],
            min_lr=ocfg.get("min_lr", 1e-6),
            warmup_epoch=ocfg.get("warmup_epoch", 5),
            step_size=ocfg.get("step_size", 300),
            gamma=ocfg.get("gamma", 0.1),
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        self.ckpt_dir = log_dir
        self.tag = lcfg.get("write_tag", "train")

        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

    def _build_anomaly_generator(self, args: Dict[str, Any]):
        cfg = args.get("anomaly_synth", {})
        synth_type = str(cfg.get("type", "cutpaste")).lower()

        if synth_type == "cutpaste":
            cutpaste_cfg = cfg.get("cutpaste", {})
            color_jitter = float(cutpaste_cfg.get("color_jitter", 0.5))
            return "cutpaste", CutPasteUnion(colorJitter=color_jitter)

        if synth_type != "ldm3d":
            raise ValueError(f"Unknown anomaly_synth.type: {synth_type}")

        dcfg = args["data"]
        lcfg = cfg.get("ldm3d", {})
        current_dataset = str(dcfg.get("dataset", "")).lower()

        # ---- choose mask dataset root (target dataset of current few-shot training) ----
        mask_dataset_root = lcfg.get("mask_dataset_root", None)
        if not mask_dataset_root:
            # Prefer explicit per-dataset roots if provided; else fallback to data.test_root
            per_ds_roots = lcfg.get("dataset_roots", {})
            mask_dataset_root = per_ds_roots.get(current_dataset, None) if isinstance(per_ds_roots, dict) else None
        if not mask_dataset_root:
            mask_dataset_root = dcfg.get("test_root", None)
        if not mask_dataset_root:
            raise ValueError(
                "LDM3D anomaly synthesis requires mask dataset root. "
                "Please set anomaly_synth.ldm3d.mask_dataset_root or anomaly_synth.ldm3d.dataset_roots.<dataset> or data.test_root."
            )

        # ---- choose LoRA path (cross-domain to preserve few-shot protocol) ----
        # Priority:
        # 1) explicit lora_path
        # 2) cross_domain_lora.<current_dataset>
        # 3) same_domain_lora.<current_dataset> (debug fallback)
        unet_lora_path = lcfg.get("lora_path", None)
        if not unet_lora_path:
            cross_map = lcfg.get("cross_domain_lora", {})
            same_map = lcfg.get("same_domain_lora", {})
            if isinstance(cross_map, dict) and current_dataset in cross_map:
                unet_lora_path = cross_map[current_dataset]
            elif isinstance(same_map, dict) and current_dataset in same_map:
                unet_lora_path = same_map[current_dataset]

        if not lcfg.get("enabled", True):
            logger.warning("anomaly_synth.type=ldm3d but anomaly_synth.ldm3d.enabled=False, falling back to CutPaste.")
            cutpaste_cfg = cfg.get("cutpaste", {})
            return "cutpaste", CutPasteUnion(colorJitter=float(cutpaste_cfg.get("color_jitter", 0.5)))

        if not lcfg.get("model_path", None):
            raise ValueError("Please set anomaly_synth.ldm3d.model_path (pretrained LDM3D model path).")
        if not unet_lora_path:
            raise ValueError(
                f"Could not resolve LDM3D LoRA path for current dataset={current_dataset}. "
                "Please set anomaly_synth.ldm3d.lora_path or cross_domain_lora mapping."
            )

        logger.info("[LDM3D] current_dataset=%s", current_dataset)
        logger.info("[LDM3D] mask_dataset_root=%s", mask_dataset_root)
        logger.info("[LDM3D] using cross-domain LoRA=%s", unet_lora_path)

        from src.utils.ldm3d_synthesis import LDM3DSDEditAnomalyGenerator

        gen = LDM3DSDEditAnomalyGenerator(
            dataset=current_dataset,
            dataset_root=str(mask_dataset_root),

            # masks
            eyecandies_splits_for_masks=tuple(lcfg.get("eyecandies_splits_for_masks", ["train", "val", "test_public"])),
            skip_empty_masks=bool(lcfg.get("skip_empty_masks", True)),

            # models
            model_path=str(lcfg["model_path"]),
            unet_lora_path=str(unet_lora_path),
            lora_rank=int(lcfg.get("lora_rank", 16)),
            lora_alpha=int(lcfg.get("lora_alpha", 16)),
            target_modules=tuple(lcfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),

            # sdedit
            resolution=lcfg.get("resolution", None),
            steps=int(lcfg.get("steps", 30)),
            strength=float(lcfg.get("strength", 0.55)),
            guidance=float(lcfg.get("guidance", 6.0)),
            use_clean_as_negative=bool(lcfg.get("use_clean_as_negative", False)),
            defect_prompt=str(lcfg.get("defect_prompt", "an object with defects")),
            clean_prompt=str(lcfg.get("clean_prompt", "a clean object")),

            # dtype/device
            bf16=bool(lcfg.get("bf16", True)),
            fp16=bool(lcfg.get("fp16", False)),
            device=lcfg.get("device", None),
            local_files_only=bool(lcfg.get("local_files_only", False)),

            # cache
            cache_root=lcfg.get("cache_root", None),
            cache_k=int(lcfg.get("cache_k", 32)),
            save_original_once=bool(lcfg.get("save_original_once", True)),
            save_meta=bool(lcfg.get("save_meta", True)),

            # mask augment
            use_elastic=bool(lcfg.get("use_elastic", True)),
            alpha=float(lcfg.get("alpha", 20.0)),
            sigma=float(lcfg.get("sigma", 6.0)),
            scale_min=float(lcfg.get("scale_min", 0.3)),
            scale_max=float(lcfg.get("scale_max", 1.0)),
            rotate_max=float(lcfg.get("rotate_max", 45.0)),
            attempts_per_mask=int(lcfg.get("attempts_per_mask", 30)),
            min_coverage=float(lcfg.get("min_coverage", 0.8)),
            use_obj_intersection_if_possible=bool(lcfg.get("use_obj_intersection_if_possible", True)),

            # normalization (keep interface explicit & configurable)
            rgb_input_norm=str(lcfg.get("rgb_input_norm", "imagenet")),
            rgb_mean=tuple(lcfg.get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=tuple(lcfg.get("rgb_std", [0.229, 0.224, 0.225])),
            depth_input_norm=str(lcfg.get("depth_input_norm", "auto")),
            depth_mean=float(lcfg.get("depth_mean", 0.0)),
            depth_std=float(lcfg.get("depth_std", 1.0)),

            on_failure=str(lcfg.get("on_failure", "return_clean")),
        )
        return "ldm3d", gen

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == "l2":
            return F.mse_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        elif self.loss_mode == "smooth_l1":
            return F.smooth_l1_loss(h.flatten(0, 1), p.flatten(0, 1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save(
            {
                "predictor_rgb": self.model.predictor_rgb.state_dict(),
                "predictor_d": self.model.predictor_d.state_dict() if self.model.predictor_d is not None else None,
                "projector": self.model.projector.state_dict() if self.model.projector else None,
                "epoch": ep,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
            self.ckpt_dir / name,
        )


    def _resolve_raw_as_single_depth(self) -> bool:
        raw_as_single_depth = bool(self.raw_as_single_depth)
        if self.anomaly_gen_kind == "ldm3d" and self.model_depth_repr == "raw":
            # Keep clean/anomaly branches consistent: both use real single-channel depth.
            raw_as_single_depth = True
        return raw_as_single_depth

    def _prepare_depths_for_anomaly_generator(self, depths):
        """
        Depths fed into the anomaly generator.

        - CutPaste: use raw sensor geometry directly.
        - LDM3D   : must use single-channel real depth [B,1,H,W].
        """
        if depths is None:
            return None
        if self.anomaly_gen_kind == "ldm3d":
            return raw_depth_batch_to_single_channel_depth(depths)
        return depths

    def _prepare_depths_for_model(self, depths):
        """
        Convert raw/generated depth to the final model representation.
        """
        if depths is None:
            return None
        return raw_depth_batch_to_model_repr(
            depths,
            depth_repr=self.model_depth_repr,
            raw_as_single_depth=self._resolve_raw_as_single_depth(),
        )

    def _synthesize_anomaly_batch(self, imgs, labels, paths, depths=None, depth_paths=None):
        """
        Returns:
            imgs_abn, depths_abn_raw

        Note:
            returned depths are still in RAW geometry space;
            conversion to depth_repr happens later.
        """
        depths_for_synth = self._prepare_depths_for_anomaly_generator(depths)

        if self.anomaly_gen_kind == "cutpaste":
            if depths_for_synth is not None:
                _, (imgs_abn, depths_abn) = self.anomaly_gen(imgs, labels, depths=depths_for_synth)
            else:
                _, imgs_abn = self.anomaly_gen(imgs, labels)
                depths_abn = None
            return imgs_abn, depths_abn

        if self.anomaly_gen_kind == "ldm3d":
            if depths_for_synth is None:
                raise ValueError("LDM3D anomaly synthesis requires RGBD batch (depths is None).")
            print("Using LDM3D anomaly generator on batch of size %d" % imgs.size(0))
            _, (imgs_abn, depths_abn) = self.anomaly_gen(
                imgs,
                labels=labels,
                depths=depths_for_synth,
                paths=paths,
                depth_paths=depth_paths,
            )
            if depths_abn is None:
                raise RuntimeError("LDM3D anomaly generator returned depths_abn=None.")
            return imgs_abn, depths_abn

        raise ValueError(f"Unknown anomaly generator kind: {self.anomaly_gen_kind}")

    def train(self):
        mp.set_start_method("spawn", force=True)
        gstep = 0

        for ep in range(self.epochs):
            logger.info("Epoch %d", ep + 1)
            self.sampler.set_epoch(ep)
            loss_m, time_m = AverageMeter(), AverageMeter()

            for itr, batch in enumerate(self.loader):
                # batch: (rgb, label, rgb_path) OR (rgb, depth, label, rgb_path, depth_path)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    imgs, labels, paths = batch
                    depths, depth_paths = None, None
                elif isinstance(batch, (list, tuple)) and len(batch) == 5:
                    imgs, depths, labels, paths, depth_paths = batch
                else:
                    blen = len(batch) if isinstance(batch, (list, tuple)) else None
                    raise ValueError(f"Unexpected batch format: {type(batch)} len={blen}. Expected 3 or 5 elements.")

                imgs = imgs.to(self.device, non_blocking=True)
                if depths is not None:
                    depths = depths.to(self.device, non_blocking=True)

                # anomaly synthesis:
                # - CutPaste: same as before
                # - LDM3D  : SDEdit-based RGBD anomaly generation with cache + deformed real masks
                imgs_abn, depths_abn_raw = self._synthesize_anomaly_batch(
                    imgs=imgs, labels=labels, paths=paths, depths=depths, depth_paths=depth_paths
                )

                depths_clean_model = self._prepare_depths_for_model(depths) if depths is not None else None
                depths_abn_model = self._prepare_depths_for_model(depths_abn_raw) if depths_abn_raw is not None else None

                def _step(use_abn: bool):
                    # FoundAD-style supervision:
                    #   clean   -> clean   (identity on the normal manifold)
                    #   anomaly -> clean   (pull anomalous features back to the normal manifold)
                    # Therefore, target features must always come from the clean RGBD sample,
                    # while the predictor input can be either clean or synthesized-anomalous.
                    x_ctx = imgs_abn if use_abn else imgs
                    d_ctx = depths_abn_model if use_abn else depths_clean_model
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        h_tgt = self.model.target_features(
                            imgs,
                            paths,
                            n_layer=self.n_layer,
                            depths=depths_clean_model,
                            depth_paths=depth_paths,
                        )
                        ctx = self.model.context_features(
                            x_ctx,
                            paths,
                            n_layer=self.n_layer,
                            depths=d_ctx,
                            depth_paths=depth_paths,
                        )
                        loss = self._loss_fn(h_tgt["rgb"], ctx["pred_rgb"])
                        if "depth" in h_tgt and "pred_depth" in ctx:
                            loss = loss + self._loss_fn(h_tgt["depth"], ctx["pred_depth"])
                        return loss

                use_abn = (np.random.rand() >= 0.5)
                (loss,), t = gpu_timer(lambda: [_step(use_abn)])

                if self.use_bf16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                grad_stats = grad_logger(self.trainable_modules.named_parameters())
                self.optimizer.zero_grad()

                loss_m.update(loss.item())
                time_m.update(t)
                gstep += 1

                if gstep % 100 == 0:
                    self._save_ckpt(ep, gstep)

                self.csv_logger.log(ep + 1, itr, loss.item(), t)

                if itr % 100 == 0:
                    mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                    logger.info(
                        "[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB (%.1fms)",
                        ep + 1, itr, loss.item(), loss_m.avg, mem_mb, time_m.avg
                    )
                    if grad_stats:
                        logger.info(
                            "    grad: [%.2e %.2e] (%.2e %.2e)",
                            grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max
                        )

            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]["lr"],
            )
            if self.scheduler is not None:
                self.scheduler.step()


def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f:
            args = yaml.safe_load(f)
    Trainer(args).train()


if __name__ == "__main__":
    main()
