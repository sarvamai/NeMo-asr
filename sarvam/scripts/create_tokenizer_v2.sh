#!/bin/bash

# take args as input
lang=$1
# remove -<lf/sf/vpp/synth> suffix from lang
lang_base=${lang%-*}

version=$2

num_tokens=$3

# hashmap of manifest files lang wise
declare -A manifest_files
manifest_files=(
    ["hi"]="/data/ASR/lhotse/manifests/train_data_shar-hi-lf/cuts.{000000..000202}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-hi-sf/cuts.{000000..000668}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-hi-vpp/cuts.{000000..000046}.jsonl.gz"
    ["en"]="/data/ASR/lhotse/manifests/train_data_shar-en-lf/cuts.{000000..000171}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-en-sf/cuts.{000000..000227}.jsonl.gz"
    ["gu"]="/data/ASR/lhotse/manifests/train_data_shar-gu-lf/cuts.{000000..000079}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-gu-sf/cuts.{000000..000192}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-gu-vpp/cuts.{000000..000015}.jsonl.gz"
    ["mr"]="/data/ASR/lhotse/manifests/train_data_shar-mr-lf/cuts.{000000..000147}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-mr-sf/cuts.{000000..000358}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-mr-vpp/cuts.{000000..000031}.jsonl.gz"
    ["kn"]="/data/ASR/lhotse/manifests/train_data_shar-kn-lf/cuts.{000000..000135}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-kn-sf/cuts.{000000..000341}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-kn-vpp/cuts.{000000..000028}.jsonl.gz"
    ["ml"]="/data/ASR/lhotse/manifests/train_data_shar-ml-lf/cuts.{000000..000068}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-ml-sf/cuts.{000000..000180}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-ml-vpp/cuts.{000000..000020}.jsonl.gz"
    ["ta"]="/data/ASR/lhotse/manifests/train_data_shar-ta-lf/cuts.{000000..000092}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-ta-sf/cuts.{000000..000131}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-ta-vpp/cuts.{000000..000049}.jsonl.gz"
    ["te"]="/data/ASR/lhotse/manifests/train_data_shar-te-lf/cuts.{000000..000101}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-te-sf/cuts.{000000..000341}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-te-vpp/cuts.{000000..000023}.jsonl.gz"
    ["bn"]="/data/ASR/lhotse/manifests/train_data_shar-bn-lf/cuts.{000000..000157}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-bn-sf/cuts.{000000..000238}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-bn-vpp/cuts.{000000..000042}.jsonl.gz"
    ["od"]="/data/ASR/lhotse/manifests/train_data_shar-od-lf/cuts.{000000..000062}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-od-sf/cuts.{000000..000183}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-od-vpp/cuts.{000000..000020}.jsonl.gz"
    ["pa"]="/data/ASR/lhotse/manifests/train_data_shar-pa-lf/cuts.{000000..000041}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-pa-sf/cuts.{000000..000041}.jsonl.gz,/data/ASR/lhotse/manifests/train_data_shar-pa-vpp/cuts.{000000..000016}.jsonl.gz"
)

all_manifests=()
for lang_item in "${!manifest_files[@]}"; do
    all_manifests+=("${manifest_files[$lang_item]}")
done

# print all manifests
echo "All manifests: ${all_manifests[*]}"

# check if lang is en or spl
if [ "$lang" != "spl" ]; then
    # manifest_file="/data/mayur_sarvam_ai/data/$version/$lang/manifests/text_manifest.txt"
    # manifest_file="/data/ASR/lhotse/manifests/train_data_shar-$lang-lf,/data/ASR/lhotse/manifests/train_data_shar-$lang-sf,/data/ASR/lhotse/manifests/train_data_shar-$lang-vpp"
    # manifest_file="/data/ASR/lhotse/manifests/train_data_shar-$lang-lf/cuts.{000000..000171}.jsonl.gz"
    # manifest_file="/data/ASR/lhotse/manifests/train_cuts-mr-lf.json"
    # manifest_file="${manifest_files[$lang]}"
    # join all manifests
    manifest_file=$(IFS=,; echo "${all_manifests[*]}")

    echo "Processing ASR text tokenizer for $lang ($version) with manifest $manifest_file and lang_base $lang_base"
        # --data_file="${manifest_file}" \

    python /home/mayur_sarvam_ai/NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest="${manifest_file}" \
        --data_root="/data/mayur_sarvam_ai/data/${version}/tokenizer/$lang_base" \
        --vocab_size=${num_tokens} \
        --tokenizer="spe" \
        --spe_type="bpe" \
        --no_lower_case \
        --spe_character_coverage=1.0 \
        --log
else
    echo "Processing special tokenizer for spl tokens"

    python /home/mayur_sarvam_ai/NeMo/scripts/speech_recognition/canary/build_canary_2_special_tokenizer.py \
        /data/mayur_sarvam_ai/data/${version}/tokenizer/spl_tokens
fi
