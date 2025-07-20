#!/bin/bash

# export WANDB_API_KEY=ad507cbace1c9be9a5df82668772e39652eb4856
export WANDB_API_KEY=60a493bcc771fa35db2cbcff6712baec220b1173

echo "Running training script..."

# HYDRA_FULL_ERROR=1 python /home/mayur_sarvam_ai/NeMo/examples/asr/speech_multitask/speech_to_text_aed.py \
#     name="canary_test" \
#     +trainer.limit_train_batches=20000 \

NCCL_DEBUG=INFO HYDRA_FULL_ERROR=1 python /home/mayur_sarvam_ai/repos/NeMo-asr/examples/speechlm2/salm_train.py \
    --config-path=/home/mayur_sarvam_ai/repos/NeMo-asr/examples/speechlm2/conf \
    --config-name=canary-gemma.yaml

