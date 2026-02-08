# Stage 2 - Training with LoRA on Single GPU (no sharding)

# Update this to your actual model path (relative to where you run the script, or absolute)
MODEL_PATH="/scratch/dakkoc25/Models/BAGEL-Base-Model"

# Set to your checkpoint path if resuming, or leave empty to start fresh
RESUME_FROM=""

# WandB configuration (set WANDB_ENABLE to False if you don't want to use it)
export WANDB_API_KEY="your_wandb_key"
WANDB_ENABLE="False"
WANDB_NAME="mathcanvas_stage2_lora"
WANDB_RUNID="0"
WANDB_RESUME="allow"
WANDB_OFFLINE="False"

# Output directories
RESULTS_DIR="./results/${WANDB_NAME}--${WANDB_RUNID}"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

torchrun \
  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
  --nproc_per_node=1 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./data/configs/stage2.yaml \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --model_path $MODEL_PATH \
  --sharding_strategy FULL_SHARD \
  --num_shard 1 \
  --cpu_offload True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 32 \
  --timestep_shift 2.0 \
  --use_flex True \
  --finetune_from_hf True \
  --auto_resume True \
  --log_every 20 \
  --save_every 2000 \
  --del_previous_state True \
  --lr 1e-4 \
  --lr_scheduler cosine \
  --min_lr 1e-7 \
  --ce_weight 0.25 \
  --mse_weight 1 \
  --warmup_steps 500 \
  --total_steps 16000 \
  --num_workers 4 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 10240 \
  --expected_num_tokens 8192 \
  --prefer_buffer_before 8192 \
  --text_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --debug_batches 3 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "all-linear" \
  --enable_wandb $WANDB_ENABLE \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID \
  --wandb_resume $WANDB_RESUME \
  --wandb_offline $WANDB_OFFLINE
