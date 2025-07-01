#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=canary-flash
#SBATCH --partition=base
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --nodelist=sarvama3-a3nodeset-3,sarvama3-a3nodeset-4,sarvama3-a3nodeset-2,sarvama3-a3nodeset-6
#SBATCH --exclusive
##SBATCH --nodelist=sarvama3-a3nodeset-6
##SBATCH --cpus-per-task=208

srun --no-container-entrypoint --container-image \
    /home/mayur_sarvam_ai/containers/canary-flash-multi-v2.sqsh --container-mounts \
    /home:/home,/data:/data \
    bash /home/mayur_sarvam_ai/NeMo/sarvam/trigger_training.sh
