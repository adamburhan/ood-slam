#!/bin/bash
#SBATCH --job-name=baseline_euroc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-lpaull
#SBATCH --array=0-19 


module load python/3.12
module load httpproxy

DATASET="euroc"
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

# staging data
DATA_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}"
LABELS_SOURCE="/scratch/adamb14/ood_slam/results/${DATASET}/orbslam2"
ORB_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/orb_images"
REPO_DIR="/home/adamb14/repos/ood-slam"

# Stage images
echo "Staging ${DATASET} images..."
for seq_dir in "${DATA_SOURCE}"/*; do
    seq_name=$(basename "$seq_dir")
    echo "Copying ${seq_name}"
    cp ${seq_dir}/*.zip ${SLURM_TMPDIR}/
done

cd ${SLURM_TMPDIR}
for f in *.zip; do
    seq_name="${f%.zip}"
    echo "Unzipping ${seq_name}"
    unzip -q "$f" -d "${seq_name}"
    rm "$f"
done

echo "Images staged to: ${SLURM_TMPDIR}"
ls ${SLURM_TMPDIR}

# Stage labels
echo "Staging ${DATASET} labels..."
for seq_dir in ${LABELS_SOURCE}/*/; do
    seq_name=$(basename "$seq_dir")
    mkdir -p ${SLURM_TMPDIR}/labels/${seq_name}
    cp ${seq_dir}/labels.csv ${SLURM_TMPDIR}/labels/${seq_name}/
done

echo "Labels staged to: ${SLURM_TMPDIR}/labels"
ls ${SLURM_TMPDIR}/labels

# Stage ORB Images (Using the fix we discussed)
echo "Staging ${DATASET} ORB images..."
cp ${ORB_SOURCE}/*.tar.gz ${SLURM_TMPDIR}/
cd ${SLURM_TMPDIR}
for f in *_orb.tar.gz; do
    tar -xzf "$f"  # Extract directly to current dir to merge folders
    rm "$f"
done

echo "Staging complete."

# Activate env
source /home/adamb14/repos/ood-slam/.venv/bin/activate
mkdir -p /scratch/adamb14/ood_slam/checkpoints

export PYTHONPATH="${REPO_DIR}/ood_slam/experiments:${PYTHONPATH}"

# Run
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