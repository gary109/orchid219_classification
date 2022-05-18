## 安裝必要套件
[OK] pytorch-21.08-py3:latest
[OK] pytorch-21.06-py3:latest

[1] git clone https://gitlab.com/gary109/orchid219_classification.git

- sudo apt-get update
- sudo apt install git-lfs
- sudo apt-get install ffmpeg libsm6 libxext6  -y
- pip install torchaudio soundfile PySoundFile jiwer wandb accelerate deepspeed
- pip install git+https://github.com/huggingface/transformers.git
- pip install git+https://github.com/huggingface/datasets.git
- git clone https://github.com/huggingface/transformers.git
- git config --global user.email "gary109@gmail.com"
- git config --global user.name "GARY"
- git config --global credential.helper store
- wandb login 2cf656515a3b158f4f603aeba63181236de2fc1b
- huggingface-cli login
- accelerate config
- accelerate test


## Table of Contents


## [Pre-Train] gary109/orchid219_pretrain_vit-mae-large
---
OMP_NUM_THREADS=1 accelerate launch run_image_classification_ViT-MAE.py \
    --dataset_name "gary109/orchid219" \
    --model_name_or_path "gary109/orchid219_pretrain_vit-mae-large" \
    --output_dir="orchid219_ft_pretrain_vit-mae-large" \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --do_train --do_eval --push_to_hub \
    --push_to_hub_model_id="orchid219_ft_pretrain_vit-mae-large" \
    --learning_rate 2e-5 \
    --num_train_epochs 80 \
    --per_device_train_batch_size 56 \
    --per_device_eval_batch_size 56 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --use_auth_token \
    --hub_token="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" \
    --seed 1337 \
    --cache_dir="Orchid219"


## [Fine-Tune] gary109/orchid219_pretrain_vit-mae-large
OMP_NUM_THREADS=1 accelerate launch run_image_classification_ViT-MAE.py \
    --dataset_name "gary109/orchid219" \
    --model_name_or_path "gary109/orchid219_pretrain_vit-mae-large" \
    --output_dir="orchid219_ft_pretrain_vit-mae-large" \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --do_train --do_eval \
    --push_to_hub \
    --push_to_hub_model_id="orchid219_ft_pretrain_vit-mae-large" \
    --learning_rate 2e-5 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --use_auth_token \
    --hub_token="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" \
    --seed 1337

## [Fine-Tune] gary109/orchid219_pretrain_vit-large-mim
---
OMP_NUM_THREADS=1 accelerate launch run_image_classification.py \
    --model_name_or_path "gary109/orchid219_pretrain_vit-large-mim" \
    --dataset_name "gary109/orchid219" \
    --output_dir="orchid219_ft_pretrain_vit-large-mim" \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --do_train --do_eval --push_to_hub \
    --push_to_hub_model_id="orchid219_ft_pretrain_vit-large-mim" \
    --hub_token="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" \
    --learning_rate 2e-5 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --use_auth_token="True" \
    --seed 1337
    
    
    
--cache_dir="Orchid219"