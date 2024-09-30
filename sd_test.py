import logging
import os
from typing import Optional

import numpy as np
import rembg
from PIL import Image
from compel import Compel
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch as th
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
from transformers import pipeline

from configs import config
from utils import clear_strings_for_prompt

device = 'cuda' if th.cuda.is_available() else 'cpu'

vae = AutoencoderKL.from_single_file(os.path.join(config.sd_model.cache_dir, config.sd_model.vae),
	                                     torch_dtype=th.float16,
	                                     cache_dir=config.sd_model.cache_dir,
	                                     use_safetensors=True,
	                                     safety_checker=None).to(device)


stablediff = StableDiffusionPipeline.from_single_file(
	os.path.join(config.sd_model.cache_dir, config.sd_model.stable_diffusion),
	torch_dtype=th.float16,
	cache_dir=config.sd_model.cache_dir,
	safety_checker=None).to(device)
stablediff.safety_checker = None
stablediff.scheduler = DPMSolverMultistepScheduler.from_config(stablediff.scheduler.config,
                                                               use_karras=True,
                                                               algorithm_type='sde-dpmsolver++')
# stablediff.scheduler = EulerDiscreteScheduler.from_config(stablediff.scheduler.config)
stablediff.set_progress_bar_config(disable=config.sd_model.disable_progress_bar)
stablediff.vae = vae
stablediff.load_lora_weights(config.sd_model.cache_dir, weight_name=config.sd_model.entity_lora)
stablediff.enable_model_cpu_offload()
compel_stablediff = Compel(tokenizer=stablediff.tokenizer, text_encoder=stablediff.text_encoder,
                           truncate_long_prompts=False)
logging.getLogger().info('Loaded Stable Diffusion')

obj_prompt = "darkest dungeon, (empty flat background)+++, {entity_name}: {entity_description}, in {place_name}: {place_description}"
negative_obj_prompt = "(close-up)+++, out of shot, (multiple objects)++, (many objects)++, not centered, duplication, repetition, monochromatic, badly drawn, (detailed background)+++"



def generate_entity(entity_name: str,
                    entity_description: str,
                    entity_type: str,
                    room_name: str,
                    room_description: str,
                    room_image: Optional[Image.Image] = None) -> str:
	try:
		# generate initial semantic-context image
		entity_name, entity_description, place_name, place_description = clear_strings_for_prompt(
			[entity_name, entity_description, room_name, room_description])
		if entity_type == 'enemy':
			entity_prompt = config.entity.enemy_prompt
			negative_prompt = config.entity.negative_enemy_prompt
		else:
			entity_prompt = obj_prompt
			negative_prompt = negative_obj_prompt
		formatted_prompt = entity_prompt.format(entity_name=entity_name,
		                                        entity_description=entity_description,
		                                        place_description=place_description, place_name=place_name)
		print(f'generate_entity {formatted_prompt=}')
		conditioning = compel_stablediff.build_conditioning_tensor(formatted_prompt)
		negative_conditioning = compel_stablediff.build_conditioning_tensor(negative_prompt)
		[conditioning, negative_conditioning] = compel_stablediff.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		entity_image = stablediff(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
		                          height=config.entity.height, width=config.entity.width,
		                          num_inference_steps=config.entity.inference_steps,
		                          generator=th.Generator(device=device).manual_seed(config.rng_seed)).images[0]
		filename = os.path.join(config.entity.save_dir, f'{entity_name}_{place_name}_wb.png')
		logging.getLogger().debug(f'generate_entity {filename=}')
		entity_image.save(filename)
		entity_image = rembg.remove(entity_image, alpha_matting=True)
		# save and return filename
		filename = os.path.join(config.entity.save_dir, f'{entity_name}_{place_name}.png')
		logging.getLogger().debug(f'generate_entity {filename=}')
		entity_image.save(filename)
		return filename
	except Exception as e:
		print(e)

generate_entity(entity_name='whispering vines',
                entity_description='creeping vines that emit soft whispers, luring adventurers closer only to ensnare them',
                entity_type='trap',
                room_name='swamp lair',
                room_description='a murky, fog-covered swamp with thick mud and overgrown vegetation',)
