import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
# import spaces
from PIL import Image

SAFETY_CHECKER = False

# Constants
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
checkpoints = {
    "1-Step" : ["sdxl_lightning_1step_unet_x0.safetensors", 1],
    "2-Step" : ["sdxl_lightning_2step_unet.safetensors", 2],
    "4-Step" : ["sdxl_lightning_4step_unet.safetensors", 4],
    "8-Step" : ["sdxl_lightning_8step_unet.safetensors", 8],
}
loaded = None

# Ensure model and scheduler are initialized in GPU-enabled function
if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda:6")


if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to("cuda:6")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda:6")
        has_nsfw_concepts = safety_checker(
            images=[images],
            clip_input=safety_checker_input.pixel_values.to("cuda:6")
        )

        return images, has_nsfw_concepts

# Function 
# @spaces.GPU(enable_queue=True)
def generate_image(prompt, ckpt):
    global loaded
    print(prompt, ckpt)

    checkpoint = checkpoints[ckpt][0]
    # num_inference_steps = checkpoints[ckpt][1]
    num_inference_steps = 4

    # base = "stabilityai/stable-diffusion-xl-base-1.0"
    #     repo = "ByteDance/SDXL-Lightning"
    #     ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
    #     unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    #     # unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    #     unet.load_state_dict(torch.load(f'models/{model_name}', map_location=device))
    #     pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
    #     pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    if loaded != num_inference_steps:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample" if num_inference_steps==1 else "epsilon")
        pipe.unet.load_state_dict(torch.load(f'/home/gunjan/unified-concept-editing/models/sdxl_master_3_pok.pt', map_location='cuda:6'))
        loaded = num_inference_steps
        
    results = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=0)

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(results.images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512))
        return images[0]
    return results.images[0]



# Gradio Interface

with gr.Blocks(css="style.css") as demo:
    gr.HTML("<h1><center>SDXL-Lightning ⚡</center></h1>")
    gr.HTML("<p><center>Lightning-fast text-to-image generation</center></p><p><center><a href='https://huggingface.co/ByteDance/SDXL-Lightning'>https://huggingface.co/ByteDance/SDXL-Lightning</a></center></p>")
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(label='Enter your prompt (English)', scale=8)
            ckpt = gr.Dropdown(label='Select inference steps',choices=['1-Step', '2-Step', '4-Step', '8-Step'], value='4-Step', interactive=True)
            submit = gr.Button(scale=1, variant='primary')
    img = gr.Image(label='SDXL-Lightning Generated Image')

    prompt.submit(fn=generate_image,
                 inputs=[prompt, ckpt],
                 outputs=img,
                 )
    submit.click(fn=generate_image,
                 inputs=[prompt, ckpt],
                 outputs=img,
                 )
    
demo.queue().launch(share=True)