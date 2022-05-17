#!/bin/bash
# set -ex
#############################################################################
echo "[AutoRun] Start"

sudo apt-get update
sudo apt install git-lfs
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install torchaudio soundfile PySoundFile jiwer wandb accelerate deepspeed

# git clone https://gitlab.com/gary109/ai-light-dance_transformers.git
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/datasets.git

# git clone https://github.com/huggingface/transformers.git
git config --global user.email "gary109@gmail.com"
git config --global user.name "GARY"
git config --global credential.helper store
wandb login 2cf656515a3b158f4f603aeba63181236de2fc1b

echo "[AutoRun] Done"

