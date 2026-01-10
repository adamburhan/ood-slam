#!/bin/bash
#SBATCH --job-name=eth3d_baselines
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=rrg-lpaull
#SBATCH --array=0-19              

module load python/3.12
module load httpproxy

# --- 1. EXPERIMENT CONFIGURATION ---
DATASET="eth3d"
SEEDS=(42 43 44 45 46)
MODES=("mean" "rgb" "orb" "combined")

# Calculate which experiment this specific job should run
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MODE_IDX=$((SLURM_ARRAY_TASK_ID % 4))

SEED=${SEEDS[$SEED_IDX]}
MODE=${MODES[$MODE_IDX]}

echo "--- STARTING TASK ${SLURM_ARRAY_TASK_ID} ---"
echo "Dataset: ${DATASET}"
echo "Mode:    ${MODE}"
echo "Seed:    ${SEED}"

# --- 2. STAGING DATA ---
DATA_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/training"
LABELS_SOURCE="/scratch/adamb14/ood_slam/results/${DATASET}/orbslam2"
ORB_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/orb_images"
REPO_DIR="/home/adamb14/repos/ood-slam"

# Stage RGB Images
echo "Staging ${DATASET} images..."
cp ${DATA_SOURCE}/*.zip ${SLURM_TMPDIR}/
cd ${SLURM_TMPDIR}
for f in *.zip; do
    unzip -q "$f"
    rm "$f"
done

# Stage Labels
echo "Staging ${DATASET} labels..."
for seq_dir in ${LABELS_SOURCE}/*/; do
    seq_name=$(basename "$seq_dir")
    mkdir -p ${SLURM_TMPDIR}/labels/${seq_name}
    cp ${seq_dir}/labels.csv ${SLURM_TMPDIR}/labels/${seq_name}/
done

# Stage ORB Images (Using the fix we discussed)
echo "Staging ${DATASET} ORB images..."
cp ${ORB_SOURCE}/*.tar.gz ${SLURM_TMPDIR}/
cd ${SLURM_TMPDIR}
for f in *_orb.tar.gz; do
    tar -xzf "$f"  # Extract directly to current dir to merge folders
    rm "$f"
done

# --- 3. RUNNING ---
source /home/adamb14/repos/ood-slam/.venv/bin/activate
mkdir -p /scratch/adamb14/ood_slam/checkpoints

export PYTHONPATH="${REPO_DIR}/ood_slam/experiments:${PYTHONPATH}"

python ${REPO_DIR}/ood_slam/experiments/simple_baselines.py \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --seed ${SEED} \
    --data_dir ${SLURM_TMPDIR} \
    --labels_dir ${SLURM_TMPDIR}/labels \
    --train_sequences ${REPO_DIR}/splits/${DATASET}_train.txt \
    --val_sequences ${REPO_DIR}/splits/${DATASET}_val.txt \
    --batch_size 32 \
    --epochs 100 \
    --lr_backbone 1e-5 \
    --lr_head 1e-3 \
    --wandb