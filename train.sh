#!/usr/bin/env bash
# Train Gemma-3-4B-it with GRPO on 2 GPUs.
#   REWARD  = cros (default) | outcome | early_answering | adding_mistakes | cir | sr
#   DATASET = aiw_survo (default) | survo | games5
# CRoS, Adding Mistakes and SR need the frozen model server (see README).
# Extra verl overrides can be appended: bash train.sh trainer.logger="['console']"
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

REWARD=${REWARD:-cros}
DATASET=${DATASET:-aiw_survo}
case $DATASET in
    aiw_survo) STEPS=300; MAX_RESP=4096 ;;
    survo)     STEPS=200; MAX_RESP=4096 ;;
    games5)    STEPS=300; MAX_RESP=8192                                # correctness counts only fully correct answers
               export BINARY_CORRECTNESS=1 FROZEN_MAX_LEN=16384 EA_MAX_TOKENS=256 ;;
    *) echo "DATASET must be aiw_survo, survo or games5"; exit 1 ;;
esac

MODEL=${MODEL:-$ROOT/models/gemma-3-4b-it}
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1}
NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
EXP=${EXP:-${REWARD}_gemma3_$DATASET}
OUT=$ROOT/checkpoints/$EXP

export PYTHONPATH=$ROOT:${PYTHONPATH:-} VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false
export FROZEN_MODEL=${FROZEN_MODEL:-$MODEL} FROZEN_URL=${FROZEN_URL:-http://localhost:8740/v1}  # path as given to `vllm serve`
AGENT=()
case $REWARD in
    outcome) ;;
    cros)    export GROUP_RANK_BONUS_KEY=cros_F GROUP_RANK_BONUS_MASK=cros_scored GROUP_RANK_BONUS_OUT=cros_bonus \
                    GROUP_RANK_BONUS_LAMBDA=${LAMBDA:-0.3} ;;                      # rank correct rollouts by F
    early_answering|adding_mistakes|cir|sr)
             AGENT=(actor_rollout_ref.rollout.agent.agent_loop_config_path=$ROOT/rewards/agent_loops.yaml
                    actor_rollout_ref.rollout.agent.default_agent_loop=$REWARD) ;;   # runs the reward's probes
    *) echo "REWARD must be cros, outcome, early_answering, adding_mistakes, cir or sr"; exit 1 ;;
esac

if [[ $REWARD =~ ^(cros|adding_mistakes|sr)$ ]]; then
python - <<'EOF' || { echo "Start the frozen model server first (see README)."; exit 1; }
import json, os, urllib.request
m = json.load(urllib.request.urlopen(os.environ["FROZEN_URL"] + "/models", timeout=10))["data"][0]
assert m["id"] == os.environ["FROZEN_MODEL"], f"the server serves {m['id']}, expected {os.environ['FROZEN_MODEL']}"
need = int(os.environ.get("FROZEN_MAX_LEN", "8192"))
assert m["max_model_len"] >= need, f"the server has --max-model-len {m['max_model_len']}, needs {need}"
EOF
fi

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$ROOT/data/$DATASET/train.parquet \
    data.val_files=$ROOT/data/$DATASET/val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=$MAX_RESP \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.seed=1 \
    custom_reward_function.path=$ROOT/rewards/$REWARD.py \
    custom_reward_function.name=compute_score \
    reward_model.num_workers=32 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm={limit_mm_per_prompt: {image: 0}}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEPS \
    trainer.project_name=cros \
    trainer.experiment_name=$EXP \
    trainer.logger="['console','wandb']" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$OUT \
    trainer.rollout_data_dir=$OUT/rollout_data \
    ${AGENT[@]+"${AGENT[@]}"} "$@"
