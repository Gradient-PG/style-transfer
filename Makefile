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

run:
	$(python) main.py

test:
	$(python) -m pytest

remove:
	$(RM) venv
