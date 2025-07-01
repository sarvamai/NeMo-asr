#!/bin/bash

export WANDB_API_KEY=ad507cbace1c9be9a5df82668772e39652eb4856

echo "Running training script..."

# HYDRA_FULL_ERROR=1 python /home/mayur_sarvam_ai/NeMo/examples/asr/speech_multitask/speech_to_text_aed.py \
#     name="canary_test" \
#     +trainer.limit_train_batches=20000 \

HYDRA_FULL_ERROR=1 python /home/mayur_sarvam_ai/NeMo/examples/asr/speech_multitask/speech_to_text_aed.py \
    --config-path=/home/mayur_sarvam_ai/NeMo/examples/asr/conf/speech_multitask \
    --config-name=fast-conformer_aed_flash \
    +trainer.limit_train_batches=20000
