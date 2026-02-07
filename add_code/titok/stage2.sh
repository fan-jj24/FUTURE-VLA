WANDB_MODE=offline \
PYTHONPATH=/your/path/to/1d-tokenizer accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --multi_gpu \
    scripts/train_titok.py \
    config=configs/training/TiTok/stage2/titok_l32.yaml \
    experiment.project="titok_l32_stage2" \
    experiment.name="titok_l32_stage2_run1" \
    experiment.output_dir="titok_l32_stage2_run1"