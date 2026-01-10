#!/bin/bash
#SBATCH --job-name=baseline_kitti
#SBATCH --output=logs/%x_%A_%a.out  # Changed to %A_%a for array logging
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=rrg-lpaull
#SBATCH --array=0-19                # <--- FIX 1: ADDED ARRAY

module load python/3.12
module load httpproxy

DATASET="kitti"
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

DATA_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/raw/sequences"
LABELS_SOURCE="/scratch/adamb14/ood_slam/results/${DATASET}/orbslam2"
ORB_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/orb_images"
REPO_DIR="/home/adamb14/repos/ood-slam"

# --- STAGE RGB IMAGES ---
echo "Staging ${DATASET} images..."
cp ${DATA_SOURCE}/*.tar.gz ${SLURM_TMPDIR}/
for f in ${SLURM_TMPDIR}/*.tar.gz; do
    tar -xzf "$f" -C ${SLURM_TMPDIR}/
    rm "$f"
done

# Detect if 'sequences' subfolder exists
if [ -d "${SLURM_TMPDIR}/sequences" ]; then
    DATA_DIR="${SLURM_TMPDIR}/sequences"
    echo "Detected 'sequences' subfolder. Setting root to ${DATA_DIR}"
else
    DATA_DIR="${SLURM_TMPDIR}"
fi

# --- STAGE LABELS ---
echo "Staging ${DATASET} labels..."
for seq_dir in ${LABELS_SOURCE}/*/; do
    seq_name=$(basename "$seq_dir")
    # Determine where labels go based on DATA_DIR
    # (Optional: Usually labels dir is passed separately, but keeping your structure)
    mkdir -p ${SLURM_TMPDIR}/labels/${seq_name}
    cp ${seq_dir}/labels.csv ${SLURM_TMPDIR}/labels/${seq_name}/
done

# --- STAGE ORB IMAGES (FIXED) ---
echo "Staging ${DATASET} ORB images..."
cp ${ORB_SOURCE}/*.tar.gz ${DATA_DIR}/   # <--- FIX 2: Copy to DATA_DIR (e.g. inside sequences/)
cd ${DATA_DIR}                            # <--- FIX 2: Extract inside DATA_DIR
for f in *_orb.tar.gz; do
    tar -xzf "$f" 
    rm "$f"
done

echo "Staging Complete. Contents of ${DATA_DIR}:"
ls ${DATA_DIR} | head -n 5

# Activate env
source /home/adamb14/repos/ood-slam/.venv/bin/activate
mkdir -p /scratch/adamb14/ood_slam/checkpoints

export PYTHONPATH="${REPO_DIR}/ood_slam/experiments:${PYTHONPATH}"

# --- RUN (FIXED DATA_DIR) ---
python ${REPO_DIR}/ood_slam/experiments/simple_baselines.py \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --seed ${SEED} \
    --data_dir ${DATA_DIR} \
    --labels_dir ${SLURM_TMPDIR}/labels \
    --train_sequences ${REPO_DIR}/splits/${DATASET}_train.txt \
    --val_sequences ${REPO_DIR}/splits/${DATASET}_val.txt \
    --batch_size 32 \
    --epochs 100 \
    --lr_backbone 1e-5 \
    --lr_head 1e-3 \
    --wandb