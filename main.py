import os 
import json
from PIL import Image 
import numpy as np 
from tqdm import tqdm
import math 
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from guidance.sd import StableDiffusion
from utils import *


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def init_model(args):
    model = StableDiffusion(args, t_range=[0.02, 0.98])
    return model
    
    
def run(args):
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Initialize model and optimizing parameters
    model = init_model(args)
    device = model.device
    guidance_scale = args.guidance_scale
    steps = args.step
    
    # Get text embeddings
    cond_embeddings = model.get_text_embeds(args.prompt)
    uncond_embeddings = model.get_text_embeds(args.negative_prompt)
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    
    if args.loss_type in ["pds"]:
        src_img = Image.open(args.src_img_path).convert("RGB")
        src_img.save(os.path.join(args.save_dir, "src_img.png"))
        
        torch_src_img = pil_to_torch(src_img).to(device)
        src_h, src_w = torch_src_img.shape[-2], torch_src_img.shape[-1]
        
        # resize such that the shortest side is 512
        l = min(src_h, src_w)
        new_h, new_w = int(512 / l * src_h), int(512 / l * src_w)
        src_img_tensor = F.interpolate(torch_src_img, (new_h, new_w), mode="bilinear")
        print("Source image shape:", src_img_tensor.shape)
        src_latents = model.encode_imgs(src_img_tensor)
        
        tgt_embeddings = model.get_text_embeds(args.edit_prompt)
        edit_embeddings = torch.cat([uncond_embeddings, tgt_embeddings])

        # Initialize latents with source images
        latents = src_latents.clone().detach().requires_grad_(True)
        
    elif args.loss_type in ["sds"]:
        # Initialize latents with random Gaussian for generation -> sds case
        latents = nn.Parameter(
            torch.randn(
                1, 4, 64, 64, 
                device=device, 
                dtype=args.precision,
            )
        )

    elif args.loss_type in ["dds"]:
        src_img = Image.open(args.src_img_path).convert("RGB")
        src_img.save(os.path.join(args.save_dir, "src_img.png"))

        torch_src_img = pil_to_torch(src_img).to(device)
        src_h, src_w = torch_src_img.shape[-2], torch_src_img.shape[-1]

        # resize such that the shortest side is 512
        l = min(src_h, src_w)
        new_h, new_w = int(512 / l * src_h), int(512 / l * src_w)
        src_img_tensor = F.interpolate(torch_src_img, (new_h, new_w), mode="bilinear")
        print("Source image shape:", src_img_tensor.shape)
        src_latents = model.encode_imgs(src_img_tensor)

        tgt_embeddings = model.get_text_embeds(args.edit_prompt)
        edit_embeddings = torch.cat([uncond_embeddings, tgt_embeddings])

        latents = src_latents.clone().detach().requires_grad_(True) # Target latents for dds
    else:
        raise ValueError("Invalid loss type")
    
    optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(steps*1.5))

    # Run optimization
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        
        if args.loss_type == "sds":
            loss = model.get_sds_loss(
                latents=latents,
                text_embeddings=text_embeddings, 
                guidance_scale=guidance_scale,
            )
            
        elif args.loss_type == "pds":
            loss = model.get_pds_loss(
                src_latents=src_latents, tgt_latents=latents, 
                src_text_embedding=text_embeddings, tgt_text_embedding=edit_embeddings,
                guidance_scale=guidance_scale, 
            )
            
        elif args.loss_type == "dds":
            loss = model.get_dds_loss(
                src_latents=src_latents, tgt_latents=latents, 
                src_text_embedding=text_embeddings, tgt_text_embedding=edit_embeddings,
                guidance_scale=guidance_scale, 
            )

        else:
            raise ValueError("Invalid loss type")
        
        (2000 * loss).backward()
            
        optimizer.step()
        scheduler.step()
        
        if step%args.log_step == 0:
            with torch.no_grad():
                img = model.decode_latents(latents)
                img_save_path = os.path.join(args.save_dir, f"{step}.png")
                torch_to_pil(img).save(img_save_path)
                print("Step: {}, Loss: {}".format(step, loss.item()))
            
    img = model.decode_latents(latents)
    if args.loss_type == "sds":
        prompt_key = args.prompt.replace(" ", "_")
    elif args.loss_type == "pds":
        prompt_key = args.edit_prompt.replace(" ", "_")
    elif args.loss_type == "dds":
        prompt_key = args.edit_prompt.replace(" ", "_")
    img_save_path = os.path.join(args.save_dir, f"{prompt_key}.png")
    torch_to_pil(img).save(img_save_path)
    print(f"Save path: {img_save_path}")
        
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str, default="low quality")
    parser.add_argument("--edit_prompt", type=str, default=None)
    parser.add_argument("--src_img_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    
    parser.add_argument("--loss_type", type=str, default="sds")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--step", type=int, default=500)
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    parser.add_argument("--log_step", type=int, default=25)
    parser.add_argument("--precision", type=str, default="fp32")
    
    return parser.parse_args()

    
def main():
    args = parse_args()
    assert args.loss_type in ["sds", "pds", "dds"], "Invalid loss type"
    if args.loss_type in ["pds", "dds"]:
        assert args.edit_prompt is not None, f"edit_prompt is required for {args.loss_type}"
        assert args.src_img_path is not None, f"src_img_path is required for {args.loss_type}"
    
    if os.path.exists(args.save_dir):
        print("[*] Save directory already exists. Overwriting...")
    else:
        os.makedirs(args.save_dir)
        
    log_opt = vars(args)
    config_path = os.path.join(args.save_dir, "run_config.yaml")
    with open(config_path, "w") as f:
        json.dump(log_opt, f, indent=4)
    
    print(f"[*] Running {args.loss_type}")
    run(args)

def auto_run_eval(loss_type="pds"):
    """
    Automatically run evaluation on all prompt-image pairs in the JSON file
    """
    with open("data/prompt_img_pairs.json", "r") as f:
        prompt_img_pairs = json.load(f)
    
    print(f"    Loss type: {loss_type}")
    save_dir = f"outputs_{loss_type}"
    if os.path.exists(save_dir):
        print(f"[*] Save directory {save_dir} already exists. Overwriting...")
    else:
        os.makedirs(save_dir)

    for prompt_key, pair in tqdm(prompt_img_pairs.items()):
        print(f"\n[*] Processing: {prompt_key}")
        print(f"    Original prompt: {pair['prompt']}")
        print(f"    Edit prompt: {pair['edit_prompt']}")
        print(f"    Image path: {pair['img_path']}")
        

        
        # Create args object for this run
        class Args:
            def __init__(self):
                self.prompt = pair['prompt']
                self.edit_prompt = pair['edit_prompt']
                self.src_img_path = pair['img_path']
                self.save_dir = save_dir
                self.loss_type = loss_type 
                self.guidance_scale = 7.5 if loss_type == "pds" else 25
                self.step = 500
                self.device = 0
                self.lr = 1e-3
                self.log_step = 25
                self.precision = "fp32"
                self.negative_prompt = "low quality"
        
        args = Args()
        
        # Save config
        log_opt = vars(args)
        # config_path = os.path.join(args.save_dir, "run_config.yaml")
        # with open(config_path, "w") as f:
        #     json.dump(log_opt, f, indent=4)
        
        print(f"[*] Running PDS for {prompt_key}")
        try:
            run(args)
            print(f"[*] Completed: {prompt_key}")
        except Exception as e:
            print(f"[ERROR] Failed to process {prompt_key}: {str(e)}")
            continue

if __name__ == "__main__":
    # For single run
    # main()
    # For runnning pds: 
    # python main.py --prompt "a cat sitting on a table" --loss_type pds --guidance_scale 7.5 --edit_prompt "a wooden cat statue sitting on a table" --src_img_path data/imgs/a_cat_sitting_on_a_table.png

    # For running sds:
    # python main.py --prompt "kyrie irving" --loss_type sds --guidance_scale 25

    # For running dds:
    # python main.py --prompt "a cat sitting on a table" --loss_type dds --guidance_scale 25 --edit_prompt "a wooden cat statue sitting on a table" --src_img_path data/imgs/a_cat_sitting_on_a_table.png

    # For batch processing all prompt-image pairs
    auto_run_eval()