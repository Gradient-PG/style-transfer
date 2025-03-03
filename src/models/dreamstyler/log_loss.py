import logging
import os
import torch
import time
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

log_dir = "log_losses"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "log_loss.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

model_path="src/models/dreamstyler/inference_style_transfer.py"
pipe = StableDiffusionPipeline.from_pretrained(model_path,torch_dtype=torch.float16).to("cuda")

logging.info(f"Model loaded from {model_path}")
logging.info(f"Model params: {sum(p.numel() for p in pipe.unet.parameters())}")

def log_metrics(prompt, num_steps=25):  #in t2i the inference_num_steps was 25
    start_time=time.time()
    image = pipe(prompt, num_inference_steps=num_steps).images[0]
    end_time=time.time()

    inference_time=end_time-start_time

    log_msg = (
        f"Prompt: {prompt}\n"
        f"Inference Time: {inference_time:.2f} sec\n"
        f"Steps: {num_steps}\n"
    )
    logging.info(log_msg)

    return image


# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def compute_clip_score(image, prompt):
    inputs = clip_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        image_features = clip_model.get_image_features(inputs["pixel_values"])
        text_features = clip_model.get_text_features(inputs["input_ids"])

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_score = (image_features @ text_features.T).cpu().numpy().item()
    logging.info(f"CLIP Score: {clip_score:.4f} for Prompt: {prompt}")
    return clip_score

#FID
def compute_fid(real_images_path, generated_images_path):
    fid_score = fid.compute_fid(real_images_path, generated_images_path, mode="clean")
    logging.info(f"FID Score: {fid_score:.4f} (Real: {real_images_path}, Generated: {generated_images_path})")
    return fid_score