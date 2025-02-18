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

setup:
	$(BASE_PY) -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

train:
	$(python) models/dreamstyler/train.py \
  --num_stages 6 \
  --train_image_path "./data/train/test0.jpg" \
  --context_prompt "A painting of forest, path, trees, in the style of {}" \
  --placeholder_token "<tst01>" \
  --output_dir "./steps/tst01" \
  --learnable_property style \
  --initializer_token painting \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 500 \
  --save_steps 100 \
  --learning_rate 0.002 \
  --lr_scheduler constant \
  --lr_warmup_steps 0

run:
	$(python) main.py

test:
	$(python) -m pytest

remove:
	$(RM) venv
