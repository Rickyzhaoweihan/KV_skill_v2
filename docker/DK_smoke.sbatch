#!/bin/bash
#SBATCH --job-name=D00DK_smoke
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# DOCKER agent GPU smoke (cluster only; EC2 uses run_eval_example.sh / repro_check.sh with docker). Runs docker/repro_check.sh inside the
# Apptainer .sif built from the Dockerfile: --nv GPU, HF cache read-only, data/outputs from the extracted transfer tars (NOT the project's outputs/).
D=/scratch/drjieliu_root/drjieliu/rickyhan/docker_build
SIF=${SIF:-$D/sif/kvskill-v2_cu130.sif}; S=/opt/singularity/4.4.1/bin/singularity; R=/opt/kvskill/KV_skill_v2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; hostname
export SINGULARITYENV_REPRO_DUMMY=1
$S exec --nv --cleanenv \
  --bind /nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home:/data/hf_home:ro \
  --bind $D/data_root/data:$R/data:ro --bind $D/data_root/outputs:$R/outputs \
  --bind /nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2/docker:/mnt/dk:ro \
  "$SIF" bash /mnt/dk/repro_check.sh
echo "${SLURM_JOB_NAME}_DONE"
