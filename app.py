import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import yaml

from src.models.dreamstyler.inference_style_transfer import style_transfer

with open("configs/demo_config.yml", 'r') as ymlfile:
	cfg = yaml.safe_load(ymlfile)

style_images = {
	v.get('name', 'Pencil'): f"./data/train/{k}/{v.get('thumbnail', '1.jpg')}" for k, v in cfg['styles'].items()
}

style_tokens = {
	v.get('name', 'Pencil'): k for k, v in cfg['styles'].items()
}


def save_img(img, token, gender, prompt, n_prompt, seed, device):
	token = style_tokens[token]
	img_width, img_height = img.size
	target_width, target_height = (512, 512)
	
	scale = max(target_width / img_width, target_height / img_height)
	new_size = (int(img_width * scale), int(img_height * scale))
	image = img.resize(new_size, Image.LANCZOS)
	
	left = (new_size[0] - target_width) / 2
	top = (new_size[1] - target_height) / 2
	right = left + target_width
	bottom = top + target_height
	image = image.crop((left, top, right, bottom))

	prompt = prompt.format(gender=gender)
	seed = int(seed)
	if seed == 0:
		seed = None
	
	return style_transfer(
		sd_path="runwayml/stable-diffusion-v1-5",
		controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
		embedding_path=f"./steps/{token}/embedding/final.bin",
		content_image_paths=None,
		content_image_raw=image,
		saveroot=None,
		prompt=prompt,
		token=token,
		num_samples=1,
		config_path="./configs/config.yml",
		returns=True,
		seed=seed,
	)[0][0]


def update_style_preview(selected_style):
	return style_images.get(selected_style, "")


with gr.Blocks() as iface:
	gr.Markdown("# Take a Picture with camera to get it in a new style")
	
	with gr.Row():
		with gr.Column():
			image_input = gr.Image(type="pil", label="Take a Picture", sources=["webcam", "upload", "clipboard"])
			
			with gr.Row(equal_height=True):
				style_preview = gr.Image(label="Preview", interactive=False, height=170,
				                         value="data/train/geometric/g1.jpg")
				
				with gr.Column():
					style_dropdown = gr.Dropdown(
						choices=list(style_tokens.keys()),
						label="Pick Transferred Style",
						value="Geometric"
					)
					gender_pick = gr.Radio(["Male face", "Female face"], label="Pick Gender", value="Female face",
					                       interactive=True)
			
			submit_button = gr.Button("Transfer style onto the image")
			
			with gr.Accordion("Advanced Options", open=False):
				prompt_input = gr.Textbox("Painting of {gender}, in the style of {{}}", label="Prompt",
				                          interactive=True)
				n_prompt_input = gr.Textbox("bad quality, low resolution, deformed, strabismus, cross-eyes",
				                            label="Negative Prompt", interactive=True)
				seed_input = gr.Number(value=0, label="Seed", interactive=True)
				device_pick = gr.Radio(["cuda:0", "cuda:1", "mbp"], label="Select Device", value="cuda:0",
				                       interactive=True)
		
		with gr.Column():
			output_image = gr.Image(label="Stylized Image")
	
	submit_button.click(
		save_img,
		inputs=[image_input, style_dropdown, gender_pick, prompt_input, n_prompt_input, seed_input, device_pick],
		outputs=output_image
	)
	style_dropdown.change(fn=update_style_preview, inputs=style_dropdown, outputs=style_preview)

if __name__ == '__main__':
	iface.launch(server_port=7863, share=True)
