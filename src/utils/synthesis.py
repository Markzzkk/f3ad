import random
import math

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import morphology
import cv2


def _depth_foreground_mask(depth: torch.Tensor, min_nonzero: float = 1e-8) -> np.ndarray:
    """Generate a binary foreground mask from depth/xyz tensor.

    Supports depth shapes [H,W], [1,H,W], [3,H,W]. Foreground is defined as non-zero.
    """
    if depth is None:
        raise ValueError("depth is None")

    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)

    d = depth.detach()
    if d.ndim == 2:
        d = d.unsqueeze(0)
    elif d.ndim != 3:
        raise ValueError(f"Unsupported depth ndim: {d.ndim}")

    d = d.float().abs()
    if d.shape[0] == 1:
        fg = (d[0] > min_nonzero)
    else:
        fg = (d.sum(dim=0) > min_nonzero)

    fg_np = fg.detach().cpu().numpy().astype(np.uint8)
    # Morphology to reduce holes and speckles.
    fg_np = morphology.closing(fg_np, morphology.square(3))
    fg_np = morphology.opening(fg_np, morphology.square(3))
    if fg_np.sum() == 0:
        # Fallback to full image to avoid breaking training.
        h, w = fg_np.shape
        fg_np = np.ones((h, w), dtype=np.uint8)
    return fg_np


def generate_target_foreground_mask(img: np.ndarray, subclass: str, depth: torch.Tensor = None) -> np.ndarray:
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    img_tensor = inv_normalize(img)
    img_tensor = torch.clamp(img_tensor, 0, 1)

    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np_uint8 = (img_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass == 'pill':
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        _, target_background_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_background_mask = (target_background_mask > 0).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['capsules']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        _, target_foreground_mask = cv2.threshold(img_np_uint8[:, :, 2], 100, 255,
                                                  cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pipe_fryum']:
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['bracket_black', 'bracket_brown', 'connector']:
        img_seg = img_np_uint8[:, :, 1]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['bracket_white', 'tubes']:
        img_seg = img_np_uint8[:, :, 2]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = target_background_mask
    elif subclass in ['metal_plate']:
        img_seg = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    else:
        # For MVTec 3D AD / EyeCandies (or any unsupported class), prefer depth/xyz foreground mask.
        if depth is not None:
            target_foreground_mask = _depth_foreground_mask(depth)
            return target_foreground_mask.astype(int)
        # Final fallback: use the full image region instead of raising and killing training.
        target_foreground_mask = np.ones_like(img_gray)

    target_foreground_mask = morphology.closing(
        target_foreground_mask, morphology.square(6))
    target_foreground_mask = morphology.opening(
        target_foreground_mask, morphology.square(6))

    if target_foreground_mask.sum() == 0:
        target_foreground_mask = np.ones_like(target_foreground_mask)

    return target_foreground_mask


class CutPaste(object):
    def __init__(self, colorJitter=0.1):
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter)

    def __call__(self, imgs, *args, **kwargs):
        return imgs, imgs


class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass, depths=None):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()
        augmented_depths = depths.clone() if depths is not None else None

        for i in range(batch_size):
            img = imgs[i]
            dep = depths[i] if depths is not None else None
            aug_img, aug_dep = self.process_image(img, subclass, dep)
            augmented_imgs[i] = aug_img
            if augmented_depths is not None:
                augmented_depths[i] = aug_dep

        if depths is None:
            return imgs, augmented_imgs
        return (imgs, depths), (augmented_imgs, augmented_depths)

    def process_image(self, img, subclass, depth=None):
        img = img.clone()
        depth_aug = depth.clone() if depth is not None else None
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass, depth=depth)  # [H, W]

        area = h * w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))

        cut_w = min(max(cut_w, 1), w)
        cut_h = min(max(cut_h, 1), h)

        if cut_w <= 0 or cut_h <= 0:
            return img, depth_aug

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y + cut_h, from_x:from_x + cut_w]
        depth_patch = None
        if depth_aug is not None:
            if depth_aug.ndim == 2:
                depth_aug = depth_aug.unsqueeze(0)
            depth_patch = depth_aug[:, from_y:from_y + cut_h, from_x:from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img, depth_aug

        valid_indices = []
        for y, x in mask_indices:
            if y + cut_h <= h and x + cut_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img, depth_aug

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        augmented[:, to_y:to_y + cut_h, to_x:to_x + cut_w] = patch

        if depth_aug is not None and depth_patch is not None:
            depth_out = depth_aug.clone()
            depth_out[:, to_y:to_y + cut_h, to_x:to_x + cut_w] = depth_patch
        else:
            depth_out = depth_aug

        return augmented, depth_out


class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass, depths=None):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()
        augmented_depths = depths.clone() if depths is not None else None

        for i in range(batch_size):
            img = imgs[i]
            dep = depths[i] if depths is not None else None
            aug_img, aug_dep = self.process_image(img, subclass, dep)
            augmented_imgs[i] = aug_img
            if augmented_depths is not None:
                augmented_depths[i] = aug_dep

        if depths is None:
            return imgs, augmented_imgs
        return (imgs, depths), (augmented_imgs, augmented_depths)

    def process_image(self, img, subclass, depth=None):
        img = img.clone()
        depth_aug = depth.clone() if depth is not None else None
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass, depth=depth)

        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))
        cut_w = min(max(cut_w, 1), w)
        cut_h = min(max(cut_h, 1), h)

        if cut_w <= 0 or cut_h <= 0:
            return img, depth_aug

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y + cut_h, from_x:from_x + cut_w]
        depth_patch = None
        if depth_aug is not None:
            if depth_aug.ndim == 2:
                depth_aug = depth_aug.unsqueeze(0)
            depth_patch = depth_aug[:, from_y:from_y + cut_h, from_x:from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(
            patch,
            angle=rot_deg,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=True,
        )
        if depth_patch is not None:
            depth_patch = TF.rotate(
                depth_patch,
                angle=rot_deg,
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=True,
            )

        _, patch_h, patch_w = patch.shape

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img, depth_aug

        valid_indices = []
        for y, x in mask_indices:
            if y + patch_h <= h and x + patch_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img, depth_aug

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        mask = torch.ones_like(patch)
        augmented = self.paste_with_mask(augmented, patch, mask, to_y, to_x)

        if depth_patch is not None and depth_aug is not None:
            depth_out = depth_aug.clone()
            depth_mask = torch.ones_like(depth_patch)
            depth_out = self.paste_with_mask(depth_out, depth_patch, depth_mask, to_y, to_x)
        else:
            depth_out = depth_aug

        return augmented, depth_out

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape

        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top:top + patch_h, left:left + patch_w]
        mask = mask.to(img_region.device)
        img_region = img_region * (1 - mask) + patch * mask
        img[:, top:top + patch_h, left:left + patch_w] = img_region

        return img


class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses, depths=None):
        batch_size = imgs.shape[0]
        augmented_imgs = imgs.clone()
        augmented_depths = depths.clone() if depths is not None else None

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)  # [1, C, H, W]
            depth_i = depths[i].unsqueeze(0) if depths is not None else None
            subclass = subclasses[i]
            if random.random() < 0.5:
                out = self.cutpaste_normal(img, subclass, depths=depth_i)
            else:
                out = self.cutpaste_scar(img, subclass, depths=depth_i)

            if depths is None:
                _, augmented = out
                augmented_imgs[i] = augmented.squeeze(0)
            else:
                _, (aug_rgb, aug_depth) = out
                augmented_imgs[i] = aug_rgb.squeeze(0)
                augmented_depths[i] = aug_depth.squeeze(0)

        if depths is None:
            return imgs, augmented_imgs
        return (imgs, depths), (augmented_imgs, augmented_depths)
