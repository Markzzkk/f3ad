import os
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, DDPMScheduler
from peft import LoraConfig

# import unified_datasets
from unified_datasets import MVTec3DGenericDataset, EyeCandiesDataset

# ----------------------------- Helper Functions -----------------------------
def add_lora_to_unet(unet, r=16, lora_alpha=16):
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, init_lora_weights="gaussian",
        target_modules=target_modules
    )
    unet.add_adapter(lora_cfg, adapter_name="anomaly_adapter")
    return [p for p in unet.parameters() if p.requires_grad]

def main():
    ap = argparse.ArgumentParser(description="Universal LDM3D Finetuning for Anomaly Detection")
    
    # Core parameters
    ap.add_argument("--dataset_type", type=str, required=True, choices=["mvtec3d", "eyecandies"], 
                    help="Choose which dataset to train on")
    
    # 路径参数
    ap.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    ap.add_argument("--model", default="/path/to/pretrained/ldm3d", help="Path to pretrained LDM3D model")
    ap.add_argument("--out", default="./lora_outputs/")
    
    # 训练超参
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    output_dir = os.path.join(args.out, f"ldm3d_lora_{args.dataset_type}")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 

    print(f"[INFO] Initializing LDM3D from {args.model}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    
    # Components
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze base
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    # Inject LoRA
    print(f"[INFO] Injecting LoRA (Rank={args.lora_rank})...")
    lora_params = add_lora_to_unet(unet, r=args.lora_rank)
    print(f"[INFO] Trainable Params: {sum(p.numel() for p in lora_params)/1e6:.2f} M")

    # Dataset Selection
    print(f"[INFO] Loading Dataset: {args.dataset_type.upper()}")
    if args.dataset_type == "mvtec3d":
        dataset = MVTec3DGenericDataset(
            root=args.data_root,
            tokenizer=tokenizer,
            resolution=args.resolution
        )
    else: # eyecandies
        dataset = EyeCandiesDataset(
            root=args.data_root,
            tokenizer=tokenizer,
            resolution=args.resolution
        )
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=len(dataloader) * args.epochs
    )

    # Training Loop
    unet.train()
    global_step = 0
    
    print(f"[INFO] Start Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # 1. Inputs
            pixel_values = batch["pixel_values"].to(device, dtype=dtype) # (B, 4, H, W)
            input_ids = batch["input_ids"].to(device)
            
            # 2. VAE Encode -> Latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 3. Add Noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Text Encode
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 5. Predict
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            # 6. Loss (Simple MSE)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # 7. Backward
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Checkpoint per epoch
        save_path = os.path.join(args.out, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)

        state_dict = {k: v for k, v in unet.state_dict().items() if "lora" in k}
        torch.save(state_dict, os.path.join(save_path, "unet_lora.pth"))

    # Final Save
    final_path = os.path.join(args.out, "final")
    os.makedirs(final_path, exist_ok=True)
    state_dict = {k: v for k, v in unet.state_dict().items() if "lora" in k}
    torch.save(state_dict, os.path.join(final_path, "unet_lora.pth"))
    print(f"[INFO] Training finished. Model saved to {final_path}")

if __name__ == "__main__":
    main()