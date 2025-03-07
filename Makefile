# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM = del /Q /F /S
    BASE_PY = python
    python = venv/Scripts/python
    pip = venv/Scripts/pip
else
    DETECTED_OS := Linux
    RM = rm -rf
    BASE_PY = python3
    python = venv/bin/python
    pip = venv/bin/pip
endif

PROMPT ?= "please provide a description"
TOKEN ?= "tst0"
CONTENT ?= "test00.jpg"

setup:
	$(BASE_PY) -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

train:
	$(python) src/models/dreamstyler/train.py \
  --num_stages 6 \
  --dataset_config "./configs/train.yml" \
  --train_image_path "./data/train/$(TOKEN).jpg" \
  --context_prompt "A painting of $(PROMPT) in the style of {}" \
  --placeholder_token "$(TOKEN)" \
  --output_dir "./steps/$(TOKEN)" \
  --learnable_property style \
  --initializer_token painting \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --resolution 512 \
  --train_batch_size 6 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 800 \
  --save_steps 100 \
  --learning_rate 0.002 \
  --lr_scheduler constant \
  --lr_warmup_steps 0

test-text:
	$(python) -m src.models.dreamstyler.inference_t2i \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path "./steps/$(TOKEN)/embedding/final.bin" \
  --prompt "Painting of $(PROMPT), in the style of {}" \
  --saveroot "./outputs/$(TOKEN)" \
  --placeholder_token "<$(TOKEN)>"

test-style:
	$(python) -m src.models.dreamstyler.inference_style_transfer \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path "./steps/$(TOKEN)/embedding/final.bin" \
  --content_image_path "./data/test/girls/$(CONTENT)" \
  --saveroot "./outputs/$(TOKEN)" \
  --token "$(TOKEN)" \
  --prompt "Painting of $(PROMPT), in the style of {}" \
  --config_path "./configs/config.yml"

test-stgrid:
	$(python) -m src.style_transfer_test_grid \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path "./steps/$(TOKEN)/embedding/final.bin" \
  --content_dir "./data/test/girls" \
  --saveroot "./outputs" \
  --token "$(TOKEN)" \
  --prompt "Painting of $(PROMPT), in the style of {}" \
  --config_path "./configs/config.yml" \
  --num_samples 3

remove:
	$(RM) venv
