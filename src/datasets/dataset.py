
import os
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torchvision
import PIL
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import numpy as np


def _depth_array_to_tensor(arr, resize: int):
    """Convert depth/xyz ndarray to float32 tensor [C,H,W], resized to (resize, resize).

    Supports:
      - HxW depth
      - HxWxC depth/xyz

    Dtype policy (generic, dataset-agnostic):
      - integer arrays (e.g. 8/16-bit PNG or TIFF): cast to float32 and normalize by dtype range
        when possible (unsigned), otherwise fall back to per-image min-max.
      - floating arrays: cast to float32 and keep numeric scale (preserve xyz geometry values).
    """
    import torch.nn.functional as F

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported depth array shape: {arr.shape}")

    arr = np.ascontiguousarray(arr)

    if np.issubdtype(arr.dtype, np.integer):
        orig_dtype = arr.dtype
        arr = arr.astype(np.float32, copy=False)
        info = np.iinfo(orig_dtype)
        if info.min >= 0 and info.max > 0:
            arr = arr / float(info.max)
        else:
            mn = float(arr.min())
            mx = float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    t = F.interpolate(
        t.unsqueeze(0),
        size=(resize, resize),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return t


def _read_depth_any(depth_path: str, resize: int):
    """Unified depth/xyz loader returning float32 tensor [C,H,W].

    Handles png/jpg/tif/tiff/npy in a dataset-agnostic way.
    """
    ext = os.path.splitext(depth_path)[1].lower()

    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(depth_path)
        except Exception:
            arr = np.array(PIL.Image.open(depth_path))
        return _depth_array_to_tensor(arr, resize)

    if ext == ".npy":
        arr = np.load(depth_path)
        return _depth_array_to_tensor(arr, resize)

    # png/jpg/etc. (preserves 16-bit depth better than torchvision ToTensor on some PIL modes)
    arr = np.array(PIL.Image.open(depth_path))
    return _depth_array_to_tensor(arr, resize)




def _read_depth_array_any(depth_path: str):
    """Read depth/xyz from png/jpg/tif/tiff/npy into a numpy array without resizing.

    Returns:
      - HxW for single-channel depth images
      - HxWx3 for xyz tiff
    """
    ext = os.path.splitext(depth_path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(depth_path)
        except Exception:
            arr = np.array(PIL.Image.open(depth_path))
        return np.asarray(arr)
    if ext == ".npy":
        return np.asarray(np.load(depth_path))
    # png/jpg/etc.
    return np.asarray(PIL.Image.open(depth_path))


def _imagenet_normalize_chw(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to a CHW float tensor."""
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)[:, None, None]
    std = torch.as_tensor(IMAGENET_STD, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean) / std


def _robust_minmax_01(x: np.ndarray, mask: np.ndarray | None = None, lo: float = 1.0, hi: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Robust min-max normalize x to [0,1] using percentiles."""
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
    """Normalize a depth map to float32 [0,1]."""
    d = np.asarray(depth)
    if np.issubdtype(d.dtype, np.integer):
        info = np.iinfo(d.dtype)
        d = d.astype(np.float32)
        if info.max > 0:
            d = d / float(info.max)
        else:
            d = _robust_minmax_01(d)
        return d
    # float depth: robust normalize for stability
    d = d.astype(np.float32, copy=False)
    return _robust_minmax_01(d)


def _infer_xyz_order(xyz: np.ndarray) -> np.ndarray:
    """Infer and reorder xyz channels to (X,Y,Z) without extra config.

    Heuristics:
      - Z (depth) channel tends to be non-negative with many zeros and larger positive values.
      - X/Y correlate with pixel coordinates (col/row).
    """
    xyz = np.asarray(xyz)
    assert xyz.ndim == 3 and xyz.shape[2] == 3
    H, W, _ = xyz.shape

    # pick Z as channel with smallest fraction of negative values; tie-breaker by largest median
    neg_fracs = []
    medians = []
    for c in range(3):
        v = xyz[..., c]
        neg_fracs.append(float((v < 0).mean()))
        medians.append(float(np.median(v[np.isfinite(v)])))
    z_idx = int(np.lexsort(([-m for m in medians], neg_fracs))[0])

    other = [i for i in range(3) if i != z_idx]

    # sample grid for correlation
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

    # X correlates with column index, Y correlates with row index
    c0, c1 = other
    v0 = xyz[..., c0][samp]
    v1 = xyz[..., c1][samp]
    corr0x = abs(_corr(v0, xx))
    corr0y = abs(_corr(v0, yy))
    corr1x = abs(_corr(v1, xx))
    corr1y = abs(_corr(v1, yy))

    # assign channel to X if it correlates more with x than y
    if corr0x + corr1y >= corr1x + corr0y:
        x_idx, y_idx = c0, c1
    else:
        x_idx, y_idx = c1, c0

    out = np.stack([xyz[..., x_idx], xyz[..., y_idx], xyz[..., z_idx]], axis=2).astype(np.float32, copy=False)
    return out


def _depth_to_pseudo_xyz(depth01: np.ndarray) -> np.ndarray:
    """Create pseudo XYZ from a single-channel depth (normalized to [0,1])."""
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
    """Compute surface normal map from XYZ. Output HxWx3 in [0,1]."""
    xyz = _infer_xyz_order(xyz_in) if (xyz_in.ndim == 3 and xyz_in.shape[2] == 3) else xyz_in.astype(np.float32)
    H, W, _ = xyz.shape
    valid = _xyz_valid_mask(xyz)

    # central differences
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

    # map [-1,1] -> [0,1]
    n01 = (n + 1.0) / 2.0

    # set invalid pixels to a neutral normal facing camera: (0,0,1) -> (0.5,0.5,1.0)
    neutral = np.array([0.5, 0.5, 1.0], dtype=np.float32)[None, None, :]
    n01 = np.where(valid[:, :, None], n01, neutral)
    return np.clip(n01, 0.0, 1.0).astype(np.float32)


def _xyz_to_hha(xyz_in: np.ndarray) -> np.ndarray:
    """Compute a lightweight HHA-like representation from XYZ.

    Output HxWx3 in [0,1]:
      - D: disparity ~ 1/Z (robust normalized)
      - H: height proxy ~ (max(Y)-Y) (robust normalized)
      - A: angle between normal and gravity (+Y) normalized by (pi/2)
    """
    xyz = _infer_xyz_order(xyz_in) if (xyz_in.ndim == 3 and xyz_in.shape[2] == 3) else xyz_in.astype(np.float32)
    valid = _xyz_valid_mask(xyz)

    X = xyz[..., 0]
    Y = xyz[..., 1]
    Z = xyz[..., 2]

    # D: disparity
    disp = 1.0 / (Z + 1e-6)
    D = _robust_minmax_01(disp, mask=valid)

    # H: height proxy (since Y increases downward in image coords, higher = smaller Y)
    Hraw = (np.nanmax(Y[valid]) - Y) if valid.any() else -Y
    Hc = _robust_minmax_01(Hraw, mask=valid)

    # A: angle between normal and gravity (+Y axis)
    # compute normal in [-1,1] (from _xyz_to_normal, invert mapping)
    n01 = _xyz_to_normal(xyz)
    n = n01 * 2.0 - 1.0
    dot = np.abs(n[..., 1])  # gravity along +Y
    dot = np.clip(dot, 0.0, 1.0)
    ang = np.arccos(dot).astype(np.float32)  # [0, pi/2]
    A = np.clip(ang / (np.pi / 2.0), 0.0, 1.0).astype(np.float32)

    hha = np.stack([D, Hc, A], axis=2).astype(np.float32)
    # invalid -> 0
    hha = np.where(valid[:, :, None], hha, 0.0).astype(np.float32)
    return hha

class RandomRotate90or270:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img

def build_base_transform(resize: int = 518):
    return [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

def build_train_transform(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    # use_random_erasing=False,
):
    ops = []

    if use_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))

    if use_vflip:
        ops.append(transforms.RandomVerticalFlip(p=0.2))

    if use_rotate90:
        ops.append(RandomRotate90or270(p=0.2))

    if use_color_jitter:
        ops.append(transforms.RandomApply(
                [transforms.ColorJitter(0.3,0.3,0.3,0.05)],
                p=0.2
            )
        )

    if use_gray:
        ops.append(transforms.RandomGrayscale(p=0.1))

    if use_blur:
        ops.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23 if resize >= 384 else 11, sigma=(0.1, 2.0))],
                p=0.2
            )
        )

    ops.extend(build_base_transform(resize))

    # if use_random_erasing:
    #     ops.append(transforms.RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3)))

    return transforms.Compose(ops)

def build_train_transform_new(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_any=0.3,
):
    candidates = []
    if use_hflip:      
        candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:      
        candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:   
        candidates.append(RandomRotate90or270(p=1.0))
    if use_color_jitter:
        candidates.append(transforms.ColorJitter(0.3,0.3,0.3,0.05))
    if use_gray:       
        candidates.append(transforms.Lambda(lambda im: im.convert("L").convert("RGB")))
    if use_blur:
        candidates.append(transforms.GaussianBlur(kernel_size=23 if resize>=384 else 11, sigma=(0.1,2.0)))

    ops = []
    if candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(candidates)],
                p=p_any
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)


def build_train_transform_staged(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_orient=0.3,
    p_appear=0.3,
):
    ops = []

    orient_candidates = []
    if use_hflip:
        orient_candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:
        orient_candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:
        orient_candidates.append(RandomRotate90or270(p=1.0))

    if orient_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(orient_candidates)],
                p=p_orient
            )
        )

    appear_candidates = []
    if use_color_jitter:
        appear_candidates.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.05))  # 确定性
    if use_gray:
        appear_candidates.append(transforms.RandomGrayscale(p=1.0))            # 确定性
    if use_blur:
        ksz = 23 if resize >= 384 else 11
        appear_candidates.append(transforms.GaussianBlur(kernel_size=ksz, sigma=(0.1, 2.0)))

    if appear_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(appear_candidates)],
                p=p_appear
            )
        )

    ops.extend(build_base_transform(resize))

    return transforms.Compose(ops)


class TrainDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root: str, resize=518, datasetname: str = "mvtec", **kwargs):
        """Train dataset.

        - For 2D datasets (mvtec/visa): keep original ImageFolder behavior.
        - For 3D/RGBD datasets (mvtec3d/eyecandies): read RGB from a dedicated subdir and
          pair with depth/xyz by the same stem.

        IMPORTANT:
        The `root` passed in training is assumed to be the *few-shot* dataset root.
        Therefore `root/train` MUST exist; otherwise we raise an error (fail fast).
        """
        train_root = os.path.join(root, "train")
        if not os.path.isdir(train_root):
            raise FileNotFoundError(
                f"[TrainDataset] Expected few-shot train directory '{train_root}' does not exist. "
                f"Please make sure you pass the few-shot dataset root (which must contain a 'train' folder)."
            )

        self.resize = resize
        self.datasetname = datasetname.lower()
        self.root = train_root

        # Augment flags (we will apply spatial aug jointly for RGB & depth in RGBD modes)
        self.use_hflip = kwargs.get("use_hflip", False)
        self.use_vflip = kwargs.get("use_vflip", False)
        self.use_rotate90 = kwargs.get("use_rotate90", False)
        self.p_orient = kwargs.get("p_orient", 0.3)
        self.p_appear = kwargs.get("p_appear", 0.3)

        # Appearance aug only for RGB
        self.use_color_jitter = kwargs.get("use_color_jitter", False)
        self.use_gray = kwargs.get("use_gray", False)
        self.use_blur = kwargs.get("use_blur", False)
        # NOTE: for RGBD experiments the dataset always returns RAW geometry.
        # depth_repr conversion now happens after anomaly synthesis / before model forward.
        self.depth_repr = "raw"

        # 2D datasets: keep original behavior
        if self.datasetname not in ("mvtec3d", "mvtec_3d", "mvtec3d_ad", "eyecandies", "eye_candies", "eye-candies"):
            super().__init__(train_root)
            self.transform = build_train_transform_staged(
                self.resize,
                use_hflip=self.use_hflip,
                use_vflip=self.use_vflip,
                use_rotate90=self.use_rotate90,
                use_color_jitter=self.use_color_jitter,
                use_gray=self.use_gray,
                use_blur=self.use_blur,
                p_orient=self.p_orient,
                p_appear=self.p_appear,
            )
            self.samples = [(path, self.classes[target]) for (path, target) in self.samples]
            print(f"Totally {len(self.samples)} will be trained..")
            return

        # -------- RGBD datasets: build our own paired samples list --------
        # Build class list from train_root subfolders
        class_dirs = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
        if not class_dirs:
            raise FileNotFoundError(f"[TrainDataset] No class subfolders found under '{train_root}'.")

        self.classes = sorted(class_dirs)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # where to find rgb/depth under each class
        # mvtec3d few-shot: train/<cls>/rgb and train/<cls>/xyz
        # eyecandies few-shot: train/<cls>/rgb and train/<cls>/depth
        if self.datasetname.startswith("mvtec3d") or self.datasetname in ("mvtec3d", "mvtec_3d", "mvtec3d_ad"):
            rgb_dirname = kwargs.get("mvtec3d_rgb_dirname", "rgb")
            depth_dirname = kwargs.get("mvtec3d_depth_dirname", "xyz")
            allowed_rgb_exts = tuple(x.lower() for x in kwargs.get("allowed_rgb_exts", [".png", ".jpg", ".jpeg"]))
            allowed_depth_exts = tuple(x.lower() for x in kwargs.get("allowed_depth_exts", [".tiff", ".tif", ".png", ".exr"]))
            self._rgbd_mode = "mvtec3d"
        else:
            rgb_dirname = kwargs.get("eyecandies_rgb_dirname", "rgb")
            depth_dirname = kwargs.get("eyecandies_depth_dirname", kwargs.get("eyecandies_target_depth_dirname", "depth"))
            allowed_rgb_exts = tuple(x.lower() for x in kwargs.get("allowed_rgb_exts", [".png", ".jpg", ".jpeg"]))
            allowed_depth_exts = tuple(x.lower() for x in kwargs.get("allowed_depth_exts", [".png", ".tiff", ".tif", ".exr"]))
            self._rgbd_mode = "eyecandies"

        self.samples = []  # (rgb_path, depth_path, target_idx)
        for cname in self.classes:
            croot = os.path.join(train_root, cname)
            rgb_root = os.path.join(croot, rgb_dirname)
            depth_root = os.path.join(croot, depth_dirname)

            if not os.path.isdir(rgb_root):
                raise FileNotFoundError(f"[TrainDataset] Missing RGB folder: '{rgb_root}'")
            if not os.path.isdir(depth_root):
                raise FileNotFoundError(f"[TrainDataset] Missing depth/xyz folder: '{depth_root}'")

            rgb_files = sorted([f for f in os.listdir(rgb_root) if os.path.isfile(os.path.join(rgb_root, f)) and os.path.splitext(f)[1].lower() in allowed_rgb_exts])
            if not rgb_files:
                raise FileNotFoundError(f"[TrainDataset] No RGB files found in '{rgb_root}'")

            # build a map of stem -> depth path (depth ext can differ)
            depth_files = [f for f in os.listdir(depth_root) if os.path.isfile(os.path.join(depth_root, f)) and os.path.splitext(f)[1].lower() in allowed_depth_exts]
            if not depth_files:
                raise FileNotFoundError(f"[TrainDataset] No depth/xyz files found in '{depth_root}'")

            depth_map = {}
            for df in depth_files:
                stem = os.path.splitext(df)[0]
                depth_map.setdefault(stem, []).append(os.path.join(depth_root, df))

            for rf in rgb_files:
                rpath = os.path.join(rgb_root, rf)
                stem = os.path.splitext(rf)[0]

                # Default exact-stem match (works for mvtec3d).
                candidate_stems = [stem]

                # EyeCandies few-shot naming:
                #   rgb   : <prefix>_image_<view>.png
                #   depth : <prefix>_depth.png
                if self._rgbd_mode == "eyecandies":
                    if "_image_" in stem:
                        prefix = stem.split("_image_", 1)[0]
                        candidate_stems = [f"{prefix}_depth", stem, prefix]
                    else:
                        # fallback if samples were renamed externally
                        candidate_stems = [stem, f"{stem}_depth"]

                matched_depths = None
                for cstem in candidate_stems:
                    if cstem in depth_map:
                        matched_depths = depth_map[cstem]
                        break

                if matched_depths is None:
                    raise FileNotFoundError(
                        f"[TrainDataset] Cannot find matching depth/xyz for RGB '{rpath}'. "
                        f"Tried stems {candidate_stems} under '{depth_root}'. "
                        f"Available depth stems (first 10): {sorted(depth_map.keys())[:10]}"
                    )

                dpath = sorted(matched_depths)[0]
                self.samples.append((rpath, dpath, self.class_to_idx[cname]))

        print(f"Totally {len(self.samples)} RGBD pairs will be trained..")

        # Build transforms:
        # - We'll apply spatial orientation aug jointly (manual) BEFORE transforms.
        # - RGB transform keeps appearance + base (resize/toTensor/normalize), with orient disabled.
        self.transform_rgb = build_train_transform_staged(
            self.resize,
            use_hflip=False,
            use_vflip=False,
            use_rotate90=False,
            use_color_jitter=self.use_color_jitter,
            use_gray=self.use_gray,
            use_blur=self.use_blur,
            p_orient=0.0,
            p_appear=self.p_appear,
        )
        # Depth transform: resize + toTensor ONLY (no ImageNet normalization)
        self.transform_depth_pil = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
        ])

    def _apply_joint_orient(self, rgb_img, depth_img):
        """Apply the SAME random orientation transform to RGB and depth/xyz."""
        ops = []
        if self.use_hflip:
            ops.append("hflip")
        if self.use_vflip:
            ops.append("vflip")
        if self.use_rotate90:
            ops.append("rot90or270")

        if not ops:
            return rgb_img, depth_img

        if random.random() >= float(self.p_orient):
            return rgb_img, depth_img

        op = random.choice(ops)
        if op == "hflip":
            rgb_img = TF.hflip(rgb_img)
            depth_img = TF.hflip(depth_img)
        elif op == "vflip":
            rgb_img = TF.vflip(rgb_img)
            depth_img = TF.vflip(depth_img)
        else:  # rot90or270
            angle = 90 if random.random() < 0.5 else 270
            rgb_img = TF.rotate(rgb_img, angle)
            depth_img = TF.rotate(depth_img, angle)

        return rgb_img, depth_img

    def _load_depth_xyz(self, depth_path: str):
        """Load RAW depth/xyz for training.

        This dataset intentionally does NOT apply depth_repr conversion.
        Any raw->(raw/hha/normal) conversion should happen later, after anomaly synthesis.
        """
        return _read_depth_any(depth_path, self.resize)
    def __getitem__(self, index):
        """Return:
        - 2D: (rgb, target, rgb_path)
        - RGBD: (rgb, depth/xyz, target, rgb_path, depth_path)
        """
        if self.datasetname not in ("mvtec3d", "mvtec_3d", "mvtec3d_ad", "eyecandies", "eye_candies", "eye-candies"):
            path_train, target = self.samples[index]
            image_train = self.loader(path_train).convert('RGB')
            image_train = self.transform(image_train)
            return image_train, target, path_train

        rgb_path, depth_path, target = self.samples[index]
        rgb_img = PIL.Image.open(rgb_path).convert("RGB")
        depth_img = self._load_depth_xyz(depth_path)

        # Joint spatial orientation augmentation
        rgb_img, depth_img = self._apply_joint_orient(rgb_img, depth_img)

        # RGB appearance + base transform
        rgb_tensor = self.transform_rgb(rgb_img)

        # Depth/XYZ tensor is already loaded as float32 [C,H,W] and resized in _load_depth_xyz
        import torch
        if torch.is_tensor(depth_img):
            depth_tensor = depth_img
        else:
            # Backward-compatible fallback (should rarely happen now)
            if depth_img.mode not in ("L", "I;16", "F"):
                depth_img = depth_img.convert("L")
            depth_tensor = self.transform_depth_pil(depth_img).float()

        return rgb_tensor, depth_tensor, target, rgb_path, depth_path

    def __len__(self):
        return len(self.samples)



class TestDataset(Dataset):

    def __init__(
        self,
        source,
        classname,
        resize=518,
        datasetname="mvtec",
        **kwargs,
    ):
        super().__init__()
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.source = source
        self.classnames_to_use = [classname]
        self.datasetname = datasetname.lower()
        self.resize = resize
        # NOTE: for evaluation the dataset also returns RAW geometry only.
        # depth_repr conversion is handled right before model forward.
        self.depth_repr = "raw"

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(self.transform_mean, self.transform_std),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )
        # depth base transform (for PIL depth)
        self.transform_depth_pil = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )

        # EyeCandies options
        self.eyecandies_test_split = kwargs.get("eyecandies_test_split", "test_public")
        self.eyecandies_view_idx = int(kwargs.get("eyecandies_view_idx", 0))
        self.eyecandies_return_individual_masks = bool(kwargs.get("eyecandies_return_individual_masks", False))

        # MVTec3D options (names under defect_type folder)
        self.mvtec3d_rgb_dirname = kwargs.get("mvtec3d_rgb_dirname", "rgb")
        self.mvtec3d_gt_dirname = kwargs.get("mvtec3d_gt_dirname", "gt")
        self.mvtec3d_depth_dirname = kwargs.get("mvtec3d_depth_dirname", "xyz")
        self.allowed_depth_exts = tuple(x.lower() for x in kwargs.get("allowed_depth_exts", [".tiff", ".tif", ".png", ".exr"]))

        # build list
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.imagesize = (3, resize, resize)

    def _load_depth_xyz(self, depth_path: str):
        """Load RAW depth/xyz for evaluation.

        This dataset intentionally does NOT apply depth_repr conversion.
        """
        return _read_depth_any(depth_path, self.resize)
    def __getitem__(self, idx):
        item = self.data_to_iterate[idx]
        # item can be (classname, anomaly, image_path, mask_path) or (classname, anomaly, image_path, mask_path, depth_path)
        if len(item) == 4:
            classname, anomaly, image_path, mask_path = item
            depth_path = None
        else:
            classname, anomaly, image_path, mask_path, depth_path = item

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        out = {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly not in ("good", "ok")),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

        if depth_path is not None:
            depth = self._load_depth_xyz(depth_path)
            out["depth"] = depth
            out["depth_path"] = depth_path

            # For EyeCandies, anomaly label should be determined by union mask (prefix_mask.png):
            # if mask is all-black -> good
            if self.datasetname.startswith("eyecandies"):
                # mask here already resized tensor; just check if any > 0
                is_anom = int((mask > 0).any().item())
                out["is_anomaly"] = is_anom
                out["anomaly"] = "defect" if is_anom else "good"

                if self.eyecandies_return_individual_masks:
                    data_root = os.path.dirname(image_path)
                    fname = os.path.basename(image_path)
                    stem = os.path.splitext(fname)[0]
                    prefix = stem.split("_image_", 1)[0] if "_image_" in stem else stem

                    part_paths = sorted([
                        os.path.join(data_root, f)
                        for f in os.listdir(data_root)
                        if f.startswith(prefix + "_") and f.endswith("_mask.png")
                        and f not in (f"{prefix}_mask.png", f"{prefix}_normals_mask.png")
                    ])
                    out["mask_parts_paths"] = part_paths

                    parts = []
                    for mp in part_paths:
                        m = PIL.Image.open(mp).convert("L")
                        parts.append(self.transform_mask(m))
                    if parts:
                        out["mask_parts"] = torch.stack(parts, dim=0)  # (K,1,H,W)
                    else:
                        out["mask_parts"] = torch.zeros((0, 1, self.resize, self.resize))

        return out

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            if self.datasetname in ("mvtec", "visa"):
                classpath = os.path.join(self.source, classname, "test")
                maskpath = os.path.join(self.source, classname, "ground_truth")
                if not os.path.isdir(classpath):
                    raise FileNotFoundError(f"[TestDataset] Missing test folder: '{classpath}'")
                if not os.path.isdir(maskpath):
                    raise FileNotFoundError(f"[TestDataset] Missing ground_truth folder: '{maskpath}'")

                anomaly_types = os.listdir(classpath)

                imgpaths_per_class[classname] = {}
                maskpaths_per_class[classname] = {}

                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]

                    if self.datasetname == "mvtec":
                        if anomaly != "good":
                            anomaly_mask_path = os.path.join(maskpath, anomaly)
                            anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                            maskpaths_per_class[classname][anomaly] = [
                                os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                            ]
                        else:
                            maskpaths_per_class[classname]["good"] = None
                    elif self.datasetname == "visa":
                        if anomaly != "ok":
                            anomaly_mask_path = os.path.join(maskpath, anomaly)
                            anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                            maskpaths_per_class[classname][anomaly] = [
                                os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                            ]
                        else:
                            maskpaths_per_class[classname]["ok"] = None

            elif self.datasetname.startswith("mvtec3d"):
                # Structure: mvtec3d_anomaly_detection/<class>/test/<defect_type>/<rgb|gt|xyz>
                class_test_root = os.path.join(self.source, classname, "test")
                if not os.path.isdir(class_test_root):
                    raise FileNotFoundError(f"[TestDataset][MVTec3D] Missing test folder: '{class_test_root}'")

                defect_types = sorted([d for d in os.listdir(class_test_root) if os.path.isdir(os.path.join(class_test_root, d))])
                if not defect_types:
                    raise FileNotFoundError(f"[TestDataset][MVTec3D] No defect types found under '{class_test_root}'")

                imgpaths_per_class[classname] = {}
                maskpaths_per_class[classname] = {}

                for defect in defect_types:
                    defect_root = os.path.join(class_test_root, defect)
                    rgb_root = os.path.join(defect_root, self.mvtec3d_rgb_dirname)
                    depth_root = os.path.join(defect_root, self.mvtec3d_depth_dirname)
                    gt_root = os.path.join(defect_root, self.mvtec3d_gt_dirname)

                    if not os.path.isdir(rgb_root):
                        raise FileNotFoundError(f"[TestDataset][MVTec3D] Missing rgb folder: '{rgb_root}'")
                    if not os.path.isdir(depth_root):
                        raise FileNotFoundError(f"[TestDataset][MVTec3D] Missing xyz folder: '{depth_root}'")

                    rgb_files = sorted([f for f in os.listdir(rgb_root) if os.path.isfile(os.path.join(rgb_root, f))])
                    if not rgb_files:
                        raise FileNotFoundError(f"[TestDataset][MVTec3D] No rgb files in '{rgb_root}'")

                    # build depth map stem->path (ext can vary)
                    depth_files = [f for f in os.listdir(depth_root) if os.path.isfile(os.path.join(depth_root, f)) and os.path.splitext(f)[1].lower() in self.allowed_depth_exts]
                    if not depth_files:
                        raise FileNotFoundError(f"[TestDataset][MVTec3D] No xyz files in '{depth_root}'")
                    depth_map = {}
                    for df in depth_files:
                        stem = os.path.splitext(df)[0]
                        depth_map.setdefault(stem, []).append(os.path.join(depth_root, df))

                    # gt map (only for defect != good)
                    gt_map = {}
                    if defect != "good":
                        if not os.path.isdir(gt_root):
                            raise FileNotFoundError(f"[TestDataset][MVTec3D] Missing gt folder: '{gt_root}' for defect '{defect}'")
                        gt_files = [f for f in os.listdir(gt_root) if os.path.isfile(os.path.join(gt_root, f))]
                        if not gt_files:
                            raise FileNotFoundError(f"[TestDataset][MVTec3D] No gt files in '{gt_root}' for defect '{defect}'")
                        for gf in gt_files:
                            stem = os.path.splitext(gf)[0]
                            gt_map.setdefault(stem, []).append(os.path.join(gt_root, gf))

                    imgpaths_per_class[classname][defect] = []
                    maskpaths_per_class[classname][defect] = [] if defect != "good" else None

                    for rf in rgb_files:
                        rpath = os.path.join(rgb_root, rf)
                        stem = os.path.splitext(rf)[0]
                        if stem not in depth_map:
                            raise FileNotFoundError(
                                f"[TestDataset][MVTec3D] Missing xyz for rgb '{rpath}'. Expected stem '{stem}' under '{depth_root}'."
                            )
                        dpath = sorted(depth_map[stem])[0]

                        if defect == "good":
                            imgpaths_per_class[classname][defect].append((rpath, dpath, None))
                        else:
                            if stem not in gt_map:
                                raise FileNotFoundError(
                                    f"[TestDataset][MVTec3D] Missing gt for rgb '{rpath}'. Expected stem '{stem}' under '{gt_root}'."
                                )
                            mpath = sorted(gt_map[stem])[0]
                            imgpaths_per_class[classname][defect].append((rpath, dpath, mpath))

            elif self.datasetname.startswith("eyecandies"):
                # Structure: Eyecandies/<class>/test_public/data (or test_private/data)
                data_root = os.path.join(self.source, classname, self.eyecandies_test_split, "data")
                if not os.path.isdir(data_root):
                    raise FileNotFoundError(f"[TestDataset][EyeCandies] Missing data folder: '{data_root}'")

                # collect rgb view files
                view_pat = f"_image_{self.eyecandies_view_idx}.png"
                rgb_files = sorted([f for f in os.listdir(data_root) if f.endswith(view_pat) and os.path.isfile(os.path.join(data_root, f))])
                if not rgb_files:
                    raise FileNotFoundError(
                        f"[TestDataset][EyeCandies] No RGB files matching '*{view_pat}' in '{data_root}'. "
                        f"Check eyecandies_view_idx={self.eyecandies_view_idx}."
                    )

                imgpaths_per_class[classname] = {"all": []}
                maskpaths_per_class[classname] = {"all": []}

                # optional individual mask cache
                for rf in rgb_files:
                    rpath = os.path.join(data_root, rf)
                    stem = os.path.splitext(rf)[0]
                    # prefix is before '_image_'
                    if "_image_" not in stem:
                        continue
                    prefix = stem.split("_image_", 1)[0]

                    dpath = os.path.join(data_root, f"{prefix}_depth.png")
                    if not os.path.isfile(dpath):
                        raise FileNotFoundError(f"[TestDataset][EyeCandies] Missing depth file: '{dpath}'")

                    # union mask path
                    union_mask = os.path.join(data_root, f"{prefix}_mask.png")
                    if not os.path.isfile(union_mask):
                        raise FileNotFoundError(f"[TestDataset][EyeCandies] Missing union mask file: '{union_mask}'")

                    # decide if anomaly by checking whether union mask is all-black
                    m_img = PIL.Image.open(union_mask).convert("L")
                    if m_img.getbbox() is not None:
                        anomaly = "defect"
                        mask_path = union_mask
                    else:
                        anomaly = "good"
                        mask_path = None

# store record: (classname, anomaly, rgb_path, mask_path, depth_path)
                    imgpaths_per_class[classname]["all"].append((rpath, dpath, mask_path, anomaly))

            else:
                raise ValueError(f"Unsupported datasetname='{self.datasetname}' for TestDataset")

        # Unroll to list
        data_to_iterate = []
        if self.datasetname in ("mvtec", "visa"):
            for classname in sorted(imgpaths_per_class.keys()):
                for anomaly in sorted(imgpaths_per_class[classname].keys()):
                    if maskpaths_per_class[classname][anomaly] is None:
                        for image_path in imgpaths_per_class[classname][anomaly]:
                            data_to_iterate.append((classname, anomaly, image_path, None))
                    else:
                        for image_path, mask_path in zip(
                            imgpaths_per_class[classname][anomaly],
                            maskpaths_per_class[classname][anomaly],
                        ):
                            data_to_iterate.append((classname, anomaly, image_path, mask_path))

        elif self.datasetname.startswith("mvtec3d"):
            for classname in sorted(imgpaths_per_class.keys()):
                for defect in sorted(imgpaths_per_class[classname].keys()):
                    for (rpath, dpath, mpath) in imgpaths_per_class[classname][defect]:
                        data_to_iterate.append((classname, defect, rpath, mpath, dpath))

        elif self.datasetname.startswith("eyecandies"):
            for classname in sorted(imgpaths_per_class.keys()):
                for (rpath, dpath, mpath, anomaly) in imgpaths_per_class[classname]["all"]:
                    data_to_iterate.append((classname, anomaly, rpath, mpath, dpath))

        return imgpaths_per_class, data_to_iterate


def build_dataloader(
    mode: str,
    root: str,
    batch_size: int,
    pin_mem: bool = True,
    **kwargs,
):
    """Return (dataset, dataloader, sampler).

    Parameters
    ----------
    mode : str
        "paired" | "mvtec"  —— decides which Dataset subclass to instantiate.
    root : str
        Root path for the dataset. For "paired" this should contain a
        ``train`` subdirectory; for "mvtec" it should be the directory that
        has all class folders (e.g. *mvtec/*).
    **kwargs : dict
        Extra arguments forwarded to the respective dataset constructor.
    """

    if mode == "train":
        dataset = TrainDataset(root=root, **kwargs)
        sampler = distributed.DistributedSampler(
            dataset,
        )
        drop_last = True
    elif mode == "test":
        dataset = TestDataset(source=root, **kwargs)
        sampler = None  # evaluation usually not distributed
        drop_last = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and mode == "test"),
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return dataset, dataloader, sampler