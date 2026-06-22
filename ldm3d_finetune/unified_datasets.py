import os
import glob
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F_nn  # Explicitly import nn.functional
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Try importing TIFF reading library (Needed for MVTec)
try:
    import tifffile
except ImportError:
    tifffile = None

# ---------------------------------------------------------
# General Utility Functions
# ---------------------------------------------------------
def normalize_depth_01(d: np.ndarray) -> np.ndarray:
    """Normalize depth map of any range to [0, 1]"""
    d_valid = d[np.isfinite(d)]
    if d_valid.size == 0:
        return np.zeros_like(d)
    
    d_min, d_max = d_valid.min(), d_valid.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(d)
    
    return (d - d_min) / (d_max - d_min)

def get_generic_prompt(is_defect: bool) -> str:
    """
    [Core Strategy] Zero-shot Generic Prompt
    Does not contain specific object names, only describes the state.
    """
    if is_defect:
        return "an object with defects"
    else:
        return "a clean object"

# ---------------------------------------------------------
# 1. MVTec 3D AD Dataset
# ---------------------------------------------------------
class MVTec3DGenericDataset(Dataset):
    def __init__(self, root, tokenizer, resolution=512):
        self.root = root
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.norm4 = T.Normalize([0.5]*4, [0.5]*4)
        
        self.samples = []
        
        if not os.path.exists(root):
            raise ValueError(f"MVTec root not found: {root}")

        categories = sorted(os.listdir(root))
        for cat in categories:
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir): continue
            
            # MVTec usually only has train and test
            sub_splits = ["train", "test"] 
            
            for s_split in sub_splits:
                split_dir = os.path.join(cat_dir, s_split)
                if not os.path.isdir(split_dir): continue
                
                defect_types = sorted(os.listdir(split_dir))
                for dtype in defect_types:
                    defect_dir = os.path.join(split_dir, dtype)
                    rgb_dir = os.path.join(defect_dir, "rgb")
                    xyz_dir = os.path.join(defect_dir, "xyz")
                    
                    if not os.path.exists(rgb_dir): continue
                    
                    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                    for rgb_path in rgb_files:
                        stem = os.path.splitext(os.path.basename(rgb_path))[0]
                        # Try multiple suffixes to find XYZ
                        xyz_path = None
                        for ext in [".tiff", ".tif"]:
                            candidate = os.path.join(xyz_dir, f"{stem}{ext}")
                            if os.path.exists(candidate):
                                xyz_path = candidate
                                break
                        
                        if xyz_path:
                            self.samples.append({
                                "rgb_path": rgb_path,
                                "xyz_path": xyz_path,
                                "is_defect": (dtype != "good")
                            })

        print(f"[MVTec3D] Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. RGB
        rgb = Image.open(item["rgb_path"]).convert("RGB")
        rgb = rgb.resize((self.resolution, self.resolution), Image.BICUBIC)
        rgb_t = T.ToTensor()(rgb)

        # 2. Depth
        if tifffile is None:
            raise ImportError("Please pip install tifffile")
        try:
            xyz = tifffile.imread(item["xyz_path"])
        except Exception as e:
            print(f"[Warning] Error reading {item['xyz_path']}: {e}")
            xyz = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)

        depth = xyz[..., 2]
        depth = np.nan_to_num(depth, nan=0.0)
        depth_norm = normalize_depth_01(depth)
        
        depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)
        depth_t = F_nn.interpolate(depth_t.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest").squeeze(0)

        # 3. Combine
        pixel_values = torch.cat([rgb_t, depth_t], dim=0)
        pixel_values = self.norm4(pixel_values)

        # 4. Prompt
        prompt = get_generic_prompt(item["is_defect"])
        inputs = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "prompt": prompt
        }

# ---------------------------------------------------------
# 2. EyeCandies Dataset (Modified: Support Train/Val/Test_Public with Mask Check)
# ---------------------------------------------------------
class EyeCandiesDataset(Dataset):
    def __init__(self, root, tokenizer, resolution=512):
        self.root = root
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.norm4 = T.Normalize([0.5]*4, [0.5]*4)
        
        self.samples = []
        
        if not os.path.exists(root):
            raise ValueError(f"EyeCandies root not found: {root}")
            
        categories = sorted(os.listdir(root))
        
        # Load all reliable splits
        target_splits = ["train", "val", "test_public"]
        
        for cat in categories:
            for split in target_splits:
                data_dir = os.path.join(root, cat, split, "data")
                if not os.path.isdir(data_dir):
                    continue
                
                # EyeCandies file structure is centered around scene ID
                # We iterate all scenes using _depth.png as anchor
                depth_files = sorted(glob.glob(os.path.join(data_dir, "*_depth.png")))
                
                for d_path in depth_files:
                    basename = os.path.basename(d_path)
                    sample_id = basename.split("_depth")[0] # e.g., "29"
                    
                    # --- Unified Defect Judgment Logic ---
                    # Regardless of whether it is train/val or test_public, verify with mask
                    is_defect = False
                    mask_path = os.path.join(data_dir, f"{sample_id}_mask.png")
                    
                    if os.path.exists(mask_path):
                        try:
                            # Use lazy loading or getbbox to speed up, avoiding full read if possible,
                            # but Image.open() is lazy by default, convert('L') + getbbox() triggers I/O
                            # Since this is init phase, safety first.
                            m = Image.open(mask_path)
                            if m.getbbox(): # Returns a box if there are non-black pixels -> Defect
                                is_defect = True
                        except:
                            pass
                    
                    # Record info
                    info_path = os.path.join(data_dir, f"{sample_id}_info_depth.yaml")
                    self.samples.append({
                        "sample_id": sample_id,
                        "dir_path": data_dir,
                        "info_path": info_path,
                        "is_defect": is_defect,
                        "split": split 
                    })
                
        print(f"[EyeCandies] Loaded {len(self.samples)} samples from {root} (Splits: {target_splits})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        dpath = item["dir_path"]
        sid = item["sample_id"]
        
        # 1. RGB (Random Lighting Augmentation)
        light_idx = random.randint(0, 5)
        rgb_p = os.path.join(dpath, f"{sid}_image_{light_idx}.png")
        if not os.path.exists(rgb_p): 
            rgb_p = os.path.join(dpath, f"{sid}_image_0.png")
        
        rgb = Image.open(rgb_p).convert("RGB")
        rgb = rgb.resize((self.resolution, self.resolution), Image.BICUBIC)
        rgb_t = T.ToTensor()(rgb)

        # 2. Depth
        depth_p = os.path.join(dpath, f"{sid}_depth.png")
        depth_raw = Image.open(depth_p)
        depth_arr = np.array(depth_raw).astype(np.float32)
        
        # Read yaml to get scale
        scale = 0.1 # default fallback
        if os.path.exists(item["info_path"]):
            try:
                with open(item["info_path"], 'r') as f:
                    info = yaml.safe_load(f)
                    scale = float(info.get('depth_scale', 0.1))
            except:
                pass
        
        depth_real = depth_arr * scale
        depth_norm = normalize_depth_01(depth_real)
        
        depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)
        depth_t = F_nn.interpolate(depth_t.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest").squeeze(0)

        # 3. Combine
        pixel_values = torch.cat([rgb_t, depth_t], dim=0)
        pixel_values = self.norm4(pixel_values)

        # 4. Prompt
        prompt = get_generic_prompt(item["is_defect"])
        inputs = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "prompt": prompt
        }