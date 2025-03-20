# DreamStyler
# Copyright (c) 2024-present NAVER Webtoon
# Apache-2.0

import os
from os.path import join as ospj
import click
import torch
import imageio
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux.processor import Processor
from .custom_pipelines import StableDiffusionControlNetPipeline

import yaml


def load_model(sd_path, controlnet_paths, controlnet_names, embedding_path, placeholder_token="<sks1>", num_stages=6):
	tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
	controlnets = [
		ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
		for path in controlnet_paths
	]
	
	placeholder_token = [f"{placeholder_token}-T{t}" for t in range(num_stages)]
	num_added_tokens = tokenizer.add_tokens(placeholder_token)
	if num_added_tokens == 0:
		raise ValueError("The tokens are already in the tokenizer")
	placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
	text_encoder.resize_token_embeddings(len(tokenizer))
	
	learned_embeds = torch.load(embedding_path)
	token_embeds = text_encoder.get_input_embeddings().weight.data
	for token, token_id in zip(placeholder_token, placeholder_token_id):
		token_embeds[token_id] = learned_embeds[token]
	
	pipeline = StableDiffusionControlNetPipeline.from_pretrained(
		sd_path,
		controlnet=controlnets,
		text_encoder=text_encoder.to(torch.float16),
		tokenizer=tokenizer,
		torch_dtype=torch.float16,
		safety_checker=None,
	)
	
	# pipeline = custom_pipelines.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
	# 	sd_path,
	# 	controlnet=controlnet,
	# 	text_encoder=text_encoder.to(torch.float16),
	# 	tokenizer=tokenizer,
	# 	torch_dtype=torch.float16,
	# 	safety_checker=None,
	# )
	pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
	# pipeline.enable_model_cpu_offload()
	
	processors = [
		Processor(name)
		for name in controlnet_names
	]
	
	return pipeline, processors


def style_transfer(
		sd_path=None,
		controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
		embedding_path=None,
		content_image_paths=None,
		content_image_raw=None,
		saveroot="./outputs",
		prompt="A painting of a city skyline, in the style of {}",
		token="sks1",
		num_stages=6,
		num_samples=5,
		resolution=512,
		config_path="config.yml",
		returns=False,
		seed=None,
):
	placeholder_token = f"<{token}>"
	if content_image_paths is None:
		content_image_paths = [content_image_raw]
	if isinstance(content_image_paths, str):
		content_image_paths = [content_image_paths]
	
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)
	
	controlnet_configs = config.get("style-tokens", {}).get(token, {}).get("controlnet", {})
	
	controlnet_paths = []
	controlnet_names = controlnet_configs.get("name", "depth_midas")
	if isinstance(controlnet_names, str):
		controlnet_names = [controlnet_names]
	
	for name in controlnet_names:
		controlnet_paths.append(config.get("controlnet_paths", {}).get(name, controlnet_path))
	weights = controlnet_configs.get("weights", [0.7])
	
	if not returns:
		os.makedirs(saveroot, exist_ok=True)
	pipeline, processors = load_model(
		sd_path,
		controlnet_paths,
		controlnet_names,
		embedding_path,
		placeholder_token,
		num_stages,
	)
	
	device = "cuda:1" if torch.cuda.is_available() else "cpu"
	
	pipeline.to(device)
	generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
	cross_attention_kwargs = {"num_stages": num_stages}
	
	content_outputs = []
	for image_path in content_image_paths:
		if isinstance(image_path, str):
			content_image = Image.open(image_path)
			content_image = content_image.resize((resolution, resolution))
		else:
			content_image = image_path
		control_image = [processor(content_image, to_pil=True) for processor in processors]
		pos_prompt = [prompt.format(f"{placeholder_token}-T{t}") for t in range(num_stages)]
		
		negative_prompt = "bad quality, low resolution, deformed, strabismus, cross-eyes"
		
		outputs = []
		torch.manual_seed(1)
		for _ in range(num_samples):
			output = pipeline(
				prompt=pos_prompt,
				negative_prompt=negative_prompt,
				num_inference_steps=30,
				generator=generator,
				image=control_image,
				controlnet_conditioning_scale=weights,
				# image=content_image,
				# control_image=control_image,
				cross_attention_kwargs=cross_attention_kwargs,
				# strength=0.2,
				# guidance_scale=10
			).images[0]
			outputs.append(output)
		content_outputs.append(outputs)
	
	if returns:
		return content_outputs
	
	outputs = np.concatenate([np.asarray(img) for img in content_outputs[0]], axis=1)
	save_path = ospj(saveroot, f"{content_image_paths[0].split('/')[-1].split('.')[0]}.png")
	imageio.imsave(save_path, outputs)


@click.command()
@click.option("--sd_path")
@click.option("--controlnet_path", default="lllyasviel/control_v11f1p_sd15_depth")
@click.option("--embedding_path")
@click.option("--content_image_path")
@click.option("--saveroot", default="./outputs")
@click.option("--prompt", default="A painting of a city skyline, in the style of {}")
@click.option("--token", default="sks1")
@click.option("--num_stages", default=6)
@click.option("--num_samples", default=5)
@click.option("--resolution", default=512)
@click.option("--seed")
@click.option("--config_path", default="config.yml")
@click.option("--returns", default=False)
def style_transfer_cli(
		sd_path,
		controlnet_path,
		embedding_path,
		content_image_path,
		saveroot,
		prompt,
		token,
		num_stages,
		num_samples,
		resolution,
		seed,
		config_path,
		returns
):
	return style_transfer(
		sd_path=sd_path,
		controlnet_path=controlnet_path,
		embedding_path=embedding_path,
		content_image_paths=content_image_path,
		content_image_raw=None,
		saveroot=saveroot,
		prompt=prompt,
		token=token,
		num_stages=num_stages,
		num_samples=num_samples,
		resolution=resolution,
		config_path=config_path,
		returns=returns,
		seed=seed,
	)


if __name__ == "__main__":
	style_transfer_cli()
