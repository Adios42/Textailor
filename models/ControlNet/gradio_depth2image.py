import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from PIL import Image
from torchvision import transforms

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def init_model(model_dir,training):
    model_name = 'control_v11f1p_sd15_depth'
    model_2 = create_model(BASE_DIR+f'/models/{model_name}.yaml').cpu()
    model_2.load_state_dict(load_state_dict(BASE_DIR+'/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model_2.load_state_dict(load_state_dict(BASE_DIR+f'/models/{model_name}.pth', location='cuda'), strict=False)
    model_2= model_2.cuda()
    for param in model_2.parameters():
        param.requires_grad = False
        
    if training:
        torch.save(model_2.state_dict(), model_dir+"/texture.ckpt")
        
        model = create_model(BASE_DIR+f'/models/{model_name}.yaml').cpu()
        model.load_state_dict(load_state_dict(model_dir+"/texture.ckpt", location='cuda'),strict=False)
        
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

    else: 
        model=None
        ddim_sampler = DDIMSampler(model_2)
    
    return model, model_2, ddim_sampler


@torch.no_grad()
def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, 
    ddim_steps, scale, seed, eta, 
    strength=1.0, detected_map=None, unknown_mask=None, save_memory=False, depth_pad=10, resampling=0, first_image=None):

    """
        unknown mask has to be an array of shape (H, W) - should has values of (0, 255)
    """
    
    with torch.no_grad():
        H, W, C = input_image.shape

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # start from noising the input image
        x0 = Image.fromarray(input_image).convert("RGB")
        x0 = np.array(x0)

        x0 = torch.from_numpy(x0).permute(2, 0, 1).float().to(model.device)
        x0 = x0.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        x0 = (x0 / 127.5) - 1.0 # NOTE input image must be normalized to [-1, 1]

        # encode input image
        # NOTE ControlNet doesn't accept the raw input image
        x0 = model.encode_first_stage(x0)
        x0 = model.get_first_stage_encoding(x0).detach()

        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False, strength=strength)
        ddim_steps = int(ddim_steps * strength) # actually DEPRECATED

        if first_image is not None:
            x0_first = Image.fromarray(first_image).convert("RGB")
            x0_first = np.array(x0_first)

            x0_first = torch.from_numpy(x0_first).permute(2, 0, 1).float().to(model.device)
            x0_first = x0_first.unsqueeze(0).repeat(num_samples, 1, 1, 1)
            x0_first = (x0_first / 127.5) - 1.0 # NOTE input image must be normalized to [-1, 1]

            # encode input image
            # NOTE ControlNet doesn't accept the raw input image
            x0_first = model.encode_first_stage(x0_first)
            x0_first = model.get_first_stage_encoding(x0_first).detach()

            ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False, strength=strength)
            ddim_steps = int(ddim_steps * strength) # actually DEPRECATED

            ddim_steps_tensor = torch.full((x0_first.shape[0],), ddim_sampler.ddim_timesteps[-1]).to(model.device)
            x_T = model.q_sample(x0_first, ddim_steps_tensor)
            

        else:
            # add noises to the maximum
            ddim_steps_tensor = torch.full((x0.shape[0],), ddim_sampler.ddim_timesteps[-1]).to(model.device)
            x_T = model.q_sample(x0, ddim_steps_tensor)
        

        # control
        if detected_map is None:
            detected_map, _ = apply_midas(resize_image(input_image, H))
            detected_map = HWC3(detected_map)

        detected_map_resized = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map_resized).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        shape = (4, H // 8, W // 8)

        if unknown_mask is not None:
            detected_map_image = Image.fromarray(detected_map.astype(np.uint8)).convert("L")
            detected_map_np = np.array(detected_map_image)
            background_mask = detected_map_np == depth_pad # bool
            background_mask = background_mask.astype(np.float32) * 255 # 0 - 255
            unknown_mask_image = unknown_mask + background_mask
            #unknown_mask is still the unknown region
            # will be used later to compose the generated region
            unknown_mask = unknown_mask.astype(np.float32)
            unknown_mask /= 255 # normalize it to 0 - 1

            compose_flag = True

            # unknown_mask_image = unknown_mask
            # unknown_mask = unknown_mask.astype(np.float32)
            # unknown_mask /= 255 # normalize it to 0 - 1

            # target: unknown region + background
            # HACK basically generate everything except known region
        else:

            detected_map_image = Image.fromarray(detected_map.astype(np.uint8)).convert("L")
            detected_map_np = np.array(detected_map_image)

            # # target: non-background region
            # unknown_mask = (detected_map_np != depth_pad).astype(np.uint8)
            # unknown_mask_image = (unknown_mask * 255.).astype(np.uint8)
            # # Image.fromarray(unknown_mask_image).save("unknown.png")

            # target: everything
            unknown_mask = np.ones_like(detected_map_np)
            unknown_mask_image = (unknown_mask * 255.).astype(np.uint8)

            compose_flag = False


        # HACK
        unknown_mask_dilate = np.copy(unknown_mask_image)
        # if resampling, do not need any pad
        if resampling>0:
            unknown_mask_dilate = cv2.dilate(unknown_mask_image, kernel=np.ones((0,0), np.uint8), iterations=2)
        else:
            unknown_mask_dilate = cv2.dilate(unknown_mask_image, kernel=np.ones((5,5), np.uint8), iterations=2)
            
        unknown_mask_dilate = Image.fromarray(unknown_mask_dilate.astype(np.uint8)).convert("L")
        unknown_mask_dilate = unknown_mask_dilate.resize((H // 8, W // 8), Image.NEAREST)
        unknown_mask_dilate = transforms.ToTensor()(unknown_mask_dilate).to(model.device)
        unknown_mask_dilate = unknown_mask_dilate.repeat(4, 1, 1)

        # HACK make sure the mask only contains 0 and 1
        try:
            assert set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist()).issubset(set([0, 1])), set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist())
        except AssertionError:
            unknown_mask_dilate = torch.round(unknown_mask_dilate)

            assert set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist()).issubset(set([0, 1])), set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist())

        if save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, x0=x0, x_T=x_T, mask=unknown_mask_dilate,
            verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            resampling=resampling
        )

        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = []
        for i in range(num_samples):
            results.append(x_samples[i])

    return results

