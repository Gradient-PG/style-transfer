import os
from os.path import join
import click
from PIL import Image
import imageio
import numpy as np

from .models.dreamstyler.inference_style_transfer import style_transfer


@click.command()
@click.option("--sd_path")
@click.option("--embedding_path")
@click.option("--content_dir", default="./content")
@click.option("--saveroot", default="./outputs")
@click.option("--prompt", default="A painting a of a city skyline, in the style of {}")
@click.option("--token", default="sks1")
@click.option("--num_stages", default=6)
@click.option("--num_samples", default=5)
@click.option("--resolution", default=512)
@click.option("--config_path", default="config.yml")
def style_grid(
		sd_path=None,
		embedding_path=None,
		content_dir=None,
		saveroot="./outputs",
		prompt="A painting of a city skyline, in the style of {}",
		token="sks1",
		num_stages=6,
		num_samples=5,
		resolution=512,
		config_path="config.yml",
):
	content_path = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
	
	images = style_transfer(
		sd_path,
		embedding_path=embedding_path,
		content_image_paths=[join(content_dir, path) for path in content_path],
		saveroot=saveroot,
		prompt=prompt, token=token,
		num_stages=num_stages, num_samples=num_samples, resolution=resolution,
		config_path=config_path, returns=True
	)
	
	def load_images(path, image_files, image_size=(resolution, resolution)):
		"""Load and resize images from the directory."""
		_images = [Image.open(os.path.join(path, img)).resize(image_size) for img in image_files]
		return _images
	
	grid_image = Image.new('RGB', ((1 + num_samples) * resolution, len(content_path) * resolution))
	
	contents = load_images(content_dir, content_path)
	for i in range(len(contents)):
		grid_image.paste(contents[i], (0, resolution * i))
		for j, out_image in enumerate(images[i]):
			grid_image.paste(out_image, (resolution + resolution * j, resolution * i))
	
	output_path = join(saveroot, f"{token}.png")
	imageio.imwrite(output_path, np.array(grid_image))


if __name__ == "__main__":
	style_grid()
