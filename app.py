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


def save_img(img, token, prompt):
	print(token)
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
	
	image.save("output_path.jpg")

	return style_transfer(
		sd_path="runwayml/stable-diffusion-v1-5",
		controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
		embedding_path=f"./steps/{token}/embedding/final.bin",
		content_image_paths=None,
		content_image_raw=image,
		saveroot=None,
		prompt="Painting of female face, in the style of {}",
		token=token,
		num_samples=1,
		config_path="./configs/config.yml",
		returns=True,
		seed=1,
	)


def update_style_preview(selected_style):
	return style_images.get(selected_style, "")


with gr.Blocks() as iface:
	gr.Markdown("# Upload Image and Add Text Overlay")
	
	with gr.Row():
		with gr.Column():
			image_input = gr.Image(type="pil", label="Upload an Image", sources=["webcam", "upload", "clipboard"])
			
			with gr.Row(equal_height=True):
				style_preview = gr.Image(label="Preview", interactive=False, height=170, value="data/train/geometric/g1.jpg")
				
				with gr.Column():
					style_dropdown = gr.Dropdown(
						choices=list(style_tokens.keys()),
						label="Pick Transferred Style",
						value="Geometric"
					)
					gender_pick = gr.Radio(["Male face", "Female face"], label="Pick Gender", value="Female face",
					                       interactive=True)
			
			submit_button = gr.Button("Add Text to Image")
			
			with gr.Accordion("Advanced Options", open=False):
				gr.Markdown("Prompt")
				gr.Markdown("Negative Prompt")
				gr.Markdown("Seed")
				gr.Markdown("Device")
		
		with gr.Column():
			output_image = gr.Image(label="Resulting Image")
	
	submit_button.click(
		save_img,
		inputs=[image_input, style_dropdown, gender_pick],
		outputs=output_image
	)
	style_dropdown.change(fn=update_style_preview, inputs=style_dropdown, outputs=style_preview)

if __name__ == '__main__':
	iface.launch(server_port=7863, share=True)
