import logging
import os
from typing import Any, List, Optional

import numpy as np
import rembg
import torch as th
from PIL import Image, ImageFilter
from PIL.ImageOps import invert
from compel import Compel
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler, UniPCMultistepScheduler, \
	EulerDiscreteScheduler, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image

from configs import config
from utils import clear_strings_for_prompt

device = 'cuda' if th.cuda.is_available() else 'cpu'


vae: Optional[AutoencoderKL] = None

controlnet_mlsd: Optional[ControlNetModel] = None
stablediff_controlnet_mlsd: Optional[StableDiffusionControlNetPipeline] = None
compel_stablediff_controlnet_mlsd: Optional[Compel] = None

stablediff: Optional[StableDiffusionPipeline] = None
compel_stablediff: Optional[Compel] = None

# controlnet_softedge: Optional[ControlNetModel] = None
# stablediff_controlnet_softedge: Optional[StableDiffusionControlNetInpaintPipeline] =None
# compel_stablediff_controlnet_softedge: Optional[Compel] = None

controlnet_inpaint: Optional[ControlNetModel] = None
stablediff_controlnet_inpaint: Optional[StableDiffusionControlNetInpaintPipeline] = None
compel_stablediff_controlnet_inpaint: Optional[Compel] = None


def load_stablediff_models(splash: Any):
	global vae
	global controlnet_mlsd, stablediff_controlnet_mlsd, compel_stablediff_controlnet_mlsd
	global stablediff, compel_stablediff
	# global controlnet_softedge, stablediff_controlnet_softedge, compel_stablediff_controlnet_softedge
	global controlnet_inpaint, stablediff_controlnet_inpaint, compel_stablediff_controlnet_inpaint
	
	vae = AutoencoderKL.from_single_file(os.path.join(config.sd_model.cache_dir, config.sd_model.vae),
	                                     torch_dtype=th.float16,
	                                     cache_dir=config.sd_model.cache_dir,
	                                     use_safetensors=True,
	                                     safety_checker=None)#.to(device)

	splash.showMessage('Loaded VAE')
	logging.getLogger('llmaker').info('Loaded VAE')
	
	controlnet_mlsd = ControlNetModel.from_pretrained(config.sd_model.controlnet_mlsd,
	                                                  torch_dtype=th.float16,
	                                                  cache_dir=config.sd_model.cache_dir,
	                                                  use_safetensors=True,
	                                                  safety_checker=None)#.to(device)

	splash.showMessage('Loaded ControlNet MLSD')
	logging.getLogger('llmaker').info('Loaded ControlNet MLSD')
	
	stablediff_controlnet_mlsd = StableDiffusionControlNetPipeline.from_single_file(
		os.path.join(config.sd_model.cache_dir, config.sd_model.stable_diffusion),
		safety_checker=None,
		cache_dir=config.sd_model.cache_dir,
		controlnet=controlnet_mlsd,
		use_safetensors=True,
		torch_dtype=th.float16)#.to(device)
	stablediff_controlnet_mlsd.safety_checker = None
	stablediff_controlnet_mlsd.scheduler = UniPCMultistepScheduler.from_config(
		stablediff_controlnet_mlsd.scheduler.config,
		use_karras=True,
		algorithm_type='sde-dpmsolver++')
	# stablediff_controlnet_mlsd.scheduler = EulerDiscreteScheduler.from_config(stablediff_controlnet_mlsd.scheduler.config)
	stablediff_controlnet_mlsd.set_progress_bar_config(disable=config.sd_model.disable_progress_bar)
	stablediff_controlnet_mlsd.vae = vae
	stablediff_controlnet_mlsd.load_lora_weights(config.sd_model.cache_dir, weight_name=config.sd_model.ambient_lora)
	stablediff_controlnet_mlsd.enable_model_cpu_offload()
	stablediff_controlnet_mlsd.enable_xformers_memory_efficient_attention()
	compel_stablediff_controlnet_mlsd = Compel(tokenizer=stablediff_controlnet_mlsd.tokenizer,
	                                           text_encoder=stablediff_controlnet_mlsd.text_encoder,
	                                           truncate_long_prompts=False)
	splash.showMessage('Loaded Stable Diffusion w/ ControlNet MLSD')
	logging.getLogger('llmaker').info('Loaded Stable Diffusion w/ ControlNet MLSD')
	
	stablediff = StableDiffusionPipeline.from_single_file(
		os.path.join(config.sd_model.cache_dir, config.sd_model.stable_diffusion),
		torch_dtype=th.float16,
		cache_dir=config.sd_model.cache_dir,
		safety_checker=None)#.to(device)
	stablediff.safety_checker = None
	stablediff.scheduler = UniPCMultistepScheduler.from_config(stablediff.scheduler.config,
	                                                               use_karras=True,
	                                                               algorithm_type='sde-dpmsolver++')
	# stablediff.scheduler = EulerDiscreteScheduler.from_config(stablediff.scheduler.config)
	stablediff.set_progress_bar_config(disable=config.sd_model.disable_progress_bar)
	stablediff.vae = vae
	stablediff.load_lora_weights(config.sd_model.cache_dir, weight_name=config.sd_model.entity_lora)
	stablediff.enable_model_cpu_offload()
	stablediff.enable_xformers_memory_efficient_attention()
	compel_stablediff = Compel(tokenizer=stablediff.tokenizer, text_encoder=stablediff.text_encoder,
	                           truncate_long_prompts=False)
	splash.showMessage('Loaded Stable Diffusion')
	logging.getLogger('llmaker').info('Loaded Stable Diffusion')

	controlnet_softedge = ControlNetModel.from_pretrained(config.sd_model.controlnet_softedge,
	                                                      torch_dtype=th.float16,
	                                                      use_safetensors=True,
	                                                      cache_dir=config.sd_model.cache_dir,
	                                                      safety_checker=None)#.to(device)
	splash.showMessage('Loaded ControlNet SoftEdge')
	logging.getLogger('llmaker').info('Loaded ControlNet SoftEdge')

	# stablediff_controlnet_softedge = StableDiffusionControlNetInpaintPipeline.from_single_file(
	# 	os.path.join(config.sd_model.cache_dir, config.sd_model.stable_diffusion),
	# 	controlnet=controlnet_softedge,
	# 	torch_dtype=th.float16,
	# 	cache_dir=config.sd_model.cache_dir,
	# 	use_safetensors=True,
	# 	num_in_channels=4,
	# 	safety_checker=None)#.to(device)
	# stablediff_controlnet_softedge.safety_checker = None
	# # stablediff_controlnet_softedge.scheduler = UniPCMultistepScheduler.from_config(
	# # 	stablediff_controlnet_softedge.scheduler.config,
	# # 	use_karras=True,
	# # 	algorithm_type='sde-dpmsolver++')
	# stablediff_controlnet_softedge.scheduler = EulerDiscreteScheduler.from_config(stablediff_controlnet_softedge.scheduler.config)
	# stablediff_controlnet_softedge.set_progress_bar_config(disable=config.sd_model.disable_progress_bar)
	# stablediff_controlnet_softedge.vae = vae
	# stablediff_controlnet_softedge.load_lora_weights(config.sd_model.cache_dir,
	#                                                  weight_name=config.sd_model.entity_lora)
	# stablediff_controlnet_softedge.enable_model_cpu_offload()
	# compel_stablediff_controlnet_softedge = Compel(tokenizer=stablediff_controlnet_softedge.tokenizer,
	#                                                text_encoder=stablediff_controlnet_softedge.text_encoder,
	#                                                truncate_long_prompts=False)
	# splash.showMessage('Loaded Stable Diffusion w/ ControlNet SoftEdge')
	# logging.getLogger('llmaker').info('Loaded Stable Diffusion w/ ControlNet SoftEdge')

	controlnet_inpaint = ControlNetModel.from_pretrained(config.sd_model.controlnet_inpaint,
	                                                     torch_dtype=th.float16,
	                                                     use_safetensors=True,
	                                                     cache_dir=config.sd_model.cache_dir,
	                                                     safety_checker=None)#.to(device)
	splash.showMessage('Loaded ControlNet InPaint')
	logging.getLogger('llmaker').info('Loaded ControlNet InPaint')

	stablediff_controlnet_inpaint = StableDiffusionControlNetInpaintPipeline.from_single_file(
		os.path.join(config.sd_model.cache_dir, config.sd_model.stable_diffusion),
		controlnet=controlnet_inpaint,
		torch_dtype=th.float16,
		cache_dir=config.sd_model.cache_dir,
		use_safetensors=True,
		num_in_channels=4,
		safety_checker=None)#.to(device)
	stablediff_controlnet_inpaint.safety_checker = None
	stablediff_controlnet_inpaint.scheduler = UniPCMultistepScheduler.from_config(
		stablediff_controlnet_inpaint.scheduler.config,
		use_karras=True,
		algorithm_type='sde-dpmsolver++')
	# stablediff_controlnet_inpaint.scheduler = EulerDiscreteScheduler.from_config(stablediff_controlnet_inpaint.scheduler.config)
	stablediff_controlnet_inpaint.set_progress_bar_config(disable=config.sd_model.disable_progress_bar)
	stablediff_controlnet_inpaint.vae = vae
	stablediff_controlnet_inpaint.load_lora_weights(config.sd_model.cache_dir,
	                                                weight_name=config.sd_model.ambient_lora)
	stablediff_controlnet_inpaint.enable_model_cpu_offload()
	stablediff_controlnet_inpaint.enable_attention_slicing()
	stablediff_controlnet_inpaint.enable_xformers_memory_efficient_attention()
	compel_stablediff_controlnet_inpaint = Compel(tokenizer=stablediff_controlnet_inpaint.tokenizer,
	                                              text_encoder=stablediff_controlnet_inpaint.text_encoder,
	                                              truncate_long_prompts=False)
	splash.showMessage('Loaded Stable Diffusion w/ ControlNet InPaint')
	logging.getLogger('llmaker').info('Loaded Stable Diffusion w/ ControlNet InPaint')


def generate_room(room_name: str,
                  room_description: str) -> str:
	control_image = invert(load_image(config.room.mask)).resize((config.room.width, config.room.height), Image.Resampling.NEAREST)
	
	try:
		room_name, room_description = clear_strings_for_prompt([room_name, room_description])
		formatted_prompt = config.room.prompt.format(room_name=room_name, room_description=room_description)
		logging.getLogger('llmaker').debug(f'generate_room {formatted_prompt=}')
		conditioning = compel_stablediff_controlnet_mlsd.build_conditioning_tensor(formatted_prompt)
		negative_conditioning = compel_stablediff_controlnet_mlsd.build_conditioning_tensor(
			config.room.negative_prompt)
		[conditioning,
		 negative_conditioning] = compel_stablediff_controlnet_mlsd.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		room_image = stablediff_controlnet_mlsd(image=control_image,
		                                        prompt_embeds=conditioning,
		                                        negative_prompt_embeds=negative_conditioning,
		                                        num_inference_steps=config.room.inference_steps,
		                                        guidance_scale=config.room.guidance_scale,
		                                        generator=th.Generator(device=device).manual_seed(
			                                        config.rng_seed)).images[0]
		filename = os.path.join(config.room.save_dir, f'{room_name}.png')
		logging.getLogger('llmaker').debug(f'generate_room {filename=}')
		room_image.save(filename)
		return filename
	except Exception as e:
		print(e)


def generate_corridor(room_names: List[str],
                      room_descriptions: List[str],
                      corridor_length: int) -> str:
	column_control_image = load_image(config.corridor.column_mask).resize((config.corridor.width, config.corridor.height), Image.Resampling.NEAREST)
	wall_control_image = invert(load_image(config.corridor.wall_mask)).resize((config.corridor.width, config.corridor.height), Image.Resampling.NEAREST)
	
	def __make_inpaint_condition(image, image_mask):
		image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
		image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
		assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
		image[image_mask > 0.5] = -1.0  # set as masked pixel
		image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
		image = th.from_numpy(image)
		return image
	
	try:
		corridor_image = Image.new('RGB', (config.corridor.width * corridor_length, config.corridor.height))
		room_names = clear_strings_for_prompt(room_names)
		room_descriptions = clear_strings_for_prompt(room_descriptions)
		# generate base tile (column)
		formatted_prompt = config.corridor.column_prompt.format(room_a_name=room_names[0],
		                                                        room_b_name=room_names[1])
		logging.getLogger('llmaker').debug(f'generate_corridor {formatted_prompt=}')
		conditioning = compel_stablediff_controlnet_mlsd.build_conditioning_tensor(formatted_prompt)
		negative_conditioning = compel_stablediff_controlnet_mlsd.build_conditioning_tensor(
			config.corridor.negative_prompt)
		[conditioning,
		 negative_conditioning] = compel_stablediff_controlnet_mlsd.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		column_image = stablediff_controlnet_mlsd(image=column_control_image,
		                                          prompt_embeds=conditioning,
		                                          negative_prompt_embeds=negative_conditioning,
		                                          num_inference_steps=config.corridor.inference_steps,
		                                          guidance_scale=config.corridor.guidance_scale,
		                                          generator=th.Generator(device=device).manual_seed(
			                                          config.rng_seed)).images[0]
		debugging_image = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_initial.png')
		logging.getLogger('llmaker').debug(f'generate_corridor {debugging_image=}')
		column_image.save(debugging_image)
		# get tileable image
		width, height = column_image.size
		midpoint = width // 2
		left_side = column_image.crop((0, 0, midpoint, height))
		right_side = column_image.crop((midpoint, 0, width, height))
		swapped_image = Image.new("RGB", (width, height))
		swapped_image.paste(right_side, (0, 0))
		swapped_image.paste(left_side, (midpoint, 0))
		debugging_image = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_swapped.png')
		logging.getLogger('llmaker').debug(f'generate_corridor {debugging_image=}')
		swapped_image.save(debugging_image)
		# build corridor image
		control_image = __make_inpaint_condition(swapped_image, wall_control_image)
		# door 1
		conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.door_prompt.format(room_name=room_descriptions[0]))
		negative_conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.negative_prompt)
		[conditioning,
		 negative_conditioning] = compel_stablediff_controlnet_inpaint.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		# generate image
		tmp_image = stablediff_controlnet_inpaint(
			prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
			num_inference_steps=config.corridor.inference_steps,
			generator=th.Generator(device=device).manual_seed(config.rng_seed),
			eta=config.corridor.eta,
			guidance_scale=config.corridor.guidance_scale / 2,
			image=swapped_image,
			mask_image=wall_control_image,
			control_image=control_image,
		).images[0]
		debugging_image = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_door1.png')
		logging.getLogger('llmaker').debug(f'generate_corridor {debugging_image=}')
		tmp_image.save(debugging_image)
		corridor_image.paste(tmp_image, (0, 0))
		# door 2
		conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.door_prompt.format(room_name=room_descriptions[1]))
		negative_conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.negative_prompt)
		[conditioning,
		 negative_conditioning] = compel_stablediff_controlnet_inpaint.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		# generate image
		tmp_image = stablediff_controlnet_inpaint(
			prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
			num_inference_steps=config.corridor.inference_steps,
			generator=th.Generator(device=device).manual_seed(config.rng_seed),
			eta=config.corridor.eta,
			guidance_scale=config.corridor.guidance_scale / 2,
			image=swapped_image,
			mask_image=wall_control_image,
			control_image=control_image,
		).images[0]
		debugging_image = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_door2.png')
		logging.getLogger('llmaker').debug(f'generate_corridor {debugging_image=}')
		tmp_image.save(debugging_image)
		corridor_image.paste(tmp_image, (corridor_image.width - width, 0))
		# ecounters images
		conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.wall_prompt.format(room_a_name=room_descriptions[0],
			                                   room_b_name=room_descriptions[1]))
		negative_conditioning = compel_stablediff_controlnet_inpaint.build_conditioning_tensor(
			config.corridor.negative_prompt)
		[conditioning,
		 negative_conditioning] = compel_stablediff_controlnet_inpaint.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		# generate image
		tmp_images = [stablediff_controlnet_inpaint(
			prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
			num_inference_steps=config.corridor.inference_steps,
			generator=th.Generator(device=device).manual_seed(config.rng_seed + i),
			eta=config.corridor.eta,
			guidance_scale=config.corridor.guidance_scale / 2,
			image=swapped_image,
			mask_image=wall_control_image,
			control_image=control_image).images[0] for i in range(corridor_length - 2)]
		for i, tmp_image in enumerate(tmp_images):
			debugging_image = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_cell{i}.png')
			logging.getLogger('llmaker').debug(f'generate_corridor {debugging_image=}')
			tmp_image.save(debugging_image)
			corridor_image.paste(tmp_image, (width * (i + 1), 0))
		filename = os.path.join(config.corridor.save_dir, f'{"-".join(room_names)}_{corridor_length}_corridor.png')
		logging.getLogger('llmaker').debug(f'generate_corridor {filename=}')
		corridor_image.save(filename)
		return filename
	except Exception as e:
		print(e)


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
			entity_prompt = config.entity.obj_prompt
			negative_prompt = config.entity.negative_obj_prompt
		formatted_prompt = entity_prompt.format(entity_name=entity_name,
		                                               entity_description=entity_description,
		                                               place_description=place_description, place_name=place_name)
		logging.getLogger('llmaker').debug(f'generate_entity {formatted_prompt=}')
		conditioning = compel_stablediff.build_conditioning_tensor(formatted_prompt)
		negative_conditioning = compel_stablediff.build_conditioning_tensor(negative_prompt)
		[conditioning, negative_conditioning] = compel_stablediff.pad_conditioning_tensors_to_same_length(
			[conditioning, negative_conditioning])
		entity_image = stablediff(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
		                          height=config.entity.height, width=config.entity.width,
		                          num_inference_steps=config.entity.inference_steps,
		                          generator=th.Generator(device=device).manual_seed(config.rng_seed)).images[0]
		filename = os.path.join(config.entity.save_dir, f'{entity_name}_{place_name}_wb.png')
		logging.getLogger('llmaker').debug(f'generate_entity {filename=}')
		entity_image.save(filename)
		entity_image = rembg.remove(entity_image, alpha_matting=True)
		# if room_image:
		# 	# prepare background image
		# 	room_image = load_image(room_image)
		# 	bw, bh = room_image.width, room_image.height
		# 	ew, eh = entity_image.width, entity_image.height
		# 	if bw < ew:
		# 		r = ew / bw
		# 		room_image = room_image.resize((int(r * bw), int(r * bh)))
		# 	elif bh < eh:
		# 		r = eh / bh
		# 		room_image = room_image.resize((int(r * bw), int(r * bh)))
		# 	room_image = room_image.crop((bw // 2 - ew // 2, 0, bw // 2 + ew // 2, eh))
		# 	# get control and mask images from semantic-context image
		# 	control_image = entity_image.convert('L').filter(ImageFilter.FIND_EDGES)
		# 	_, _, _, mask_image = entity_image.split()
		# 	# generate image-context image
		# 	entity_full_img_context = stablediff_controlnet_softedge(
		# 		prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
		# 		num_inference_steps=config.entity.inference_steps,
		# 		eta=config.entity.eta,
		# 		guidance_scale=config.entity.guidance_scale,
		# 		image=room_image,
		# 		mask_image=mask_image,
		# 		control_image=control_image,
		# 		generator=th.Generator(device=device).manual_seed(config.rng_seed)).images[0]
		# 	# remove existing background using the mask image again
		# 	entity_img_context_arr = np.array(entity_full_img_context)
		# 	mask_arr = np.array(mask_image)
		# 	entity_img_context_arr[mask_arr == 0, :] = 0
		# 	entity_img_context_arr = np.concatenate([entity_img_context_arr, np.expand_dims(mask_arr, -1)], axis=-1)
		# 	entity_image = Image.fromarray(entity_img_context_arr.astype(np.uint8))
		
		# save and return filename
		filename = os.path.join(config.entity.save_dir, f'{entity_name}_{place_name}.png')
		logging.getLogger('llmaker').debug(f'generate_entity {filename=}')
		entity_image.save(filename)
		return filename
	except Exception as e:
		print(e)
