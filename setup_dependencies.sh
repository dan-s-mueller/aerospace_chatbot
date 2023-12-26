!rm -rf .venv
!python3 -m venv .venv
!source .venv/bin/activate
%pip install ipykernel datasets trl peft uuid pandas evaluate transformers bitsandbytes torch huggingface_hub accelerate ipywidgets scipy
%pip freeze > requirements.txt