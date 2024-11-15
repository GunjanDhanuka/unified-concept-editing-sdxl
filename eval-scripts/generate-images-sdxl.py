from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from PIL import Image
import pandas as pd
import argparse
import os

def generate_images(model_name, prompts_path, save_path, device='cuda:0', guidance_scale=7.5, 
                   image_size=1024, ddim_steps=100, num_samples=10, from_case=0, till_case=1000000):
    
    # Load the SDXL pipeline
    if model_name == 'original':
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    else:
        # Load custom model
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        # unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        unet.load_state_dict(torch.load(f'models/{model_name}', map_location=device))
        pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    # Optional: Enable memory efficient attention
    # pipeline.enable_xformers_memory_efficient_attention()
    
    # Read prompts from CSV
    df = pd.read_csv(prompts_path)
    
    # Create output directory
    folder_path = f'{save_path}/{model_name.replace("diffusers-","").replace(".pt","")}'
    os.makedirs(folder_path, exist_ok=True)

    # Generate images for each prompt
    for _, row in df.iterrows():
        case_number = row.case_number
        if not (case_number >= from_case and case_number <= till_case):
            continue

        prompt = str(row.prompt)
        seed = row.evaluation_seed
        
        # Set the generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate multiple samples for the same prompt
        for sample_idx in range(num_samples):
            # Generate the image
            image = pipeline(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=0,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
            
            # Save the image
            image.save(f"{folder_path}/{case_number}_{sample_idx}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using SDXL Pipeline')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=1024)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    args = parser.parse_args()
    
    generate_images(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        till_case=args.till_case
    )