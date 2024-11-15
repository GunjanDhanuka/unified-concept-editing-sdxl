import numpy as np
import torch
import pandas as pd
from PIL import Image
import argparse
import os, json
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import copy

def edit_model_sdxl(
    ldm_stable, 
    old_text_, 
    new_text_, 
    retain_text_=None, 
    lamb=1,
    erase_scale=0.1,
    preserve_scale=0.1,
    technique='tensor'
):
    # Get cross-attention layers
    ca_layers = []
    sub_nets = ldm_stable.unet.named_children()
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    # Initialize projection matrices
    projection_matrices = []
    og_matrices = []
    
    # Get to_v and to_k matrices
    for l in ca_layers:
        projection_matrices.append(l.to_v)
        og_matrices.append(copy.deepcopy(l.to_v))
        projection_matrices.append(l.to_k)
        og_matrices.append(copy.deepcopy(l.to_k))

    # Reset parameters
    num_layers = len(ca_layers)
    for idx in range(num_layers):
        ca_layers[idx].to_v = copy.deepcopy(og_matrices[idx*2])
        ca_layers[idx].to_k = copy.deepcopy(og_matrices[idx*2+1])
        projection_matrices[idx*2] = ca_layers[idx].to_v
        projection_matrices[idx*2+1] = ca_layers[idx].to_k

    # Process text inputs
    old_texts = [old_text_] if isinstance(old_text_, str) else old_text_
    new_texts = [t if t != '' else ' ' for t in new_text_]
    ret_texts = retain_text_ if retain_text_ is not None else ['']
    retain = retain_text_ is not None

    # Edit each projection matrix
    for layer_num, proj_matrix in enumerate(projection_matrices):
        with torch.no_grad():
            # Initialize matrices
            hidden_dim = proj_matrix.weight.shape[0]
            embed_dim = proj_matrix.weight.shape[1]
            mat1 = lamb * proj_matrix.weight
            mat2 = lamb * torch.eye(embed_dim, device=proj_matrix.weight.device)

            # Process each text pair
            for old_text, new_text in zip(old_texts, new_texts):
                # Get embeddings
                text_input = ldm_stable.tokenizer_2(
                    [old_text, new_text], 
                    padding="max_length",
                    max_length=ldm_stable.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(ldm_stable.device)
                
                # Get embeddings from both encoders
                text_input1 = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(ldm_stable.device)

                embeddings1 = ldm_stable.text_encoder(
                    text_input1.input_ids,
                    output_hidden_states=True
                ).hidden_states[-2]  # 768 dim

                embeddings2 = ldm_stable.text_encoder_2(
                    text_input.input_ids,
                    output_hidden_states=True
                ).hidden_states[-2]  # 1280 dim

                embeddings = torch.cat([embeddings1, embeddings2], dim=-1)  # 2048 dim
                # import pdb; pdb.set_trace()
                # Process token indices
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = embeddings[0]
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                new_emb = embeddings[1]
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]

                # mask = text_input.attention_mask
                # old_last_token = mask[0].sum().item() - 2
                # new_last_token = mask[1].sum().item() - 2
                # max_tokens = max(old_last_token, new_last_token)

                # # Extract relevant embeddings
                # old_emb = embeddings[0, :old_last_token]
                # new_emb = embeddings[1, :new_last_token]

                # Project embeddings
                old_proj = torch.mm(old_emb, proj_matrix.weight.t())
                new_proj = torch.mm(new_emb, proj_matrix.weight.t())

                if technique == 'tensor':
                    # Calculate projection and residual
                    u = old_proj / old_proj.norm(dim=-1, keepdim=True)
                    new_proj_scalar = (u * new_proj).sum()
                    target = new_proj - new_proj_scalar * u
                else:
                    target = new_proj

                # Update matrices
                old_emb_flat = old_emb.view(-1, embed_dim)
                target_flat = target.view(-1, hidden_dim)
                
                mat1 += erase_scale * torch.mm(target_flat.t(), old_emb_flat)
                mat2 += erase_scale * torch.mm(old_emb_flat.t(), old_emb_flat)

            # Handle preservation if specified
            if retain:
                for text in ret_texts:
                    # Get embeddings from both encoders
                    text_input1 = ldm_stable.tokenizer(
                        [text, text],
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(ldm_stable.device)

                    text_input2 = ldm_stable.tokenizer_2(
                        [text, text],
                        padding="max_length",
                        max_length=ldm_stable.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(ldm_stable.device)

                    emb1 = ldm_stable.text_encoder(
                        text_input1.input_ids,
                        output_hidden_states=True
                    ).hidden_states[-2][0, :text_input1.attention_mask[0].sum()-2]

                    emb2 = ldm_stable.text_encoder_2(
                        text_input2.input_ids,
                        output_hidden_states=True
                    ).hidden_states[-2][0, :text_input2.attention_mask[0].sum()-2]

                    emb = torch.cat([emb1, emb2], dim=-1)  # Concatenate to 2048 dim
                    proj = torch.mm(emb, proj_matrix.weight.t())

                    emb_flat = emb.view(-1, embed_dim)
                    proj_flat = proj.view(-1, hidden_dim)
                    
                    mat1 += preserve_scale * torch.mm(proj_flat.t(), emb_flat)
                    mat2 += preserve_scale * torch.mm(emb_flat.t(), emb_flat)

            # Update projection matrix
            # import pdb; pdb.set_trace()
            mat1 = mat1.float()
            mat2 = mat2.float()
            proj_matrix.weight = torch.nn.Parameter(torch.mm(mat1, torch.inverse(mat2)).half())

    return ldm_stable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Edit SDXL model to modify concept associations'
    )
    
    parser.add_argument('--concepts', help='concepts to edit', type=str, required=True)
    parser.add_argument('--guided_concepts', help='target concepts', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='concepts to preserve', type=str, default=None)
    parser.add_argument('--technique', help='editing technique (replace/tensor)', type=str, default='replace')
    parser.add_argument('--device', help='cuda device', type=str, default='0')
    parser.add_argument('--preserve_scale', help='preservation weight', type=float, default=0.1)
    parser.add_argument('--erase_scale', help='erasure weight', type=float, default=1)
    
    args = parser.parse_args()
    
    # Load model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = f'cuda:{args.device}'
    # ldm_stable = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    ldm_stable = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
    ldm_stable.scheduler = EulerDiscreteScheduler.from_config(ldm_stable.scheduler.config, timestep_spacing="trailing")
    
    # Process concepts
    concepts = [c.strip() for c in args.concepts.split(',')]
    new_concepts = [c.strip() for c in args.guided_concepts.split(',')] if args.guided_concepts else [' ' for _ in concepts]
    preserve_concepts = [c.strip() for c in args.preserve_concepts.split(',')] if args.preserve_concepts else None
    
    print(f"Editing: {concepts} → {new_concepts}")
    print(f"Preserving: {preserve_concepts}")
    
    # Edit model
    ldm_stable = edit_model_sdxl(
        ldm_stable=ldm_stable,
        old_text_=concepts,
        new_text_=new_concepts,
        retain_text_=preserve_concepts,
        erase_scale=args.erase_scale,
        preserve_scale=args.preserve_scale,
        technique=args.technique
    )
    
    # Save outputs
    output_name = f"sdxl_edited_{'_'.join(concepts)}"
    if args.guided_concepts:
        output_name += f"_to_{'_'.join(new_concepts)}"
    if args.preserve_concepts:
        output_name += "_with_preservation"
        
    os.makedirs('models', exist_ok=True)
    os.makedirs('info', exist_ok=True)
    
    torch.save(ldm_stable.unet.state_dict(), f'models/{output_name}.pt')
    
    with open(f'info/{output_name}.json', 'w') as f:
        json.dump({
            'edited_concepts': concepts,
            'target_concepts': new_concepts,
            'preserved_concepts': preserve_concepts,
            'technique': args.technique,
            'erase_scale': args.erase_scale,
            'preserve_scale': args.preserve_scale
        }, f)