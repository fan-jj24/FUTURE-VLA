WANDB_MODE=offline \
PYTHONPATH=/your/path/to/1d-tokenizer accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --multi_gpu \
    scripts/train_titok.py \
    config=configs/training/TiTok/stage1/titok_l32.yaml \
    experiment.project="titok_l32_stage1" \
    experiment.name="titok_l32_stage1_run1" \
    experiment.output_dir="titok_l32_stage1_run1"