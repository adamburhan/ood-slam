#!/bin/bash
#SBATCH --job-name=tartan_base
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=rrg-lpaull
#SBATCH --array=0-19

module load python/3.12
module load httpproxy

DATASET="tartanair"
SEEDS=(42 43 44 45 46)
MODES=("mean" "rgb" "orb" "combined")

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MODE_IDX=$((SLURM_ARRAY_TASK_ID % 4))
SEED=${SEEDS[$SEED_IDX]}
MODE=${MODES[$MODE_IDX]}

echo "--- TASK ${SLURM_ARRAY_TASK_ID}: ${MODE} / Seed ${SEED} ---"

DATA_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}_zipped"
LABELS_SOURCE="/scratch/adamb14/ood_slam/results/${DATASET}/orbslam2"
ORB_SOURCE="/scratch/adamb14/ood_slam/datasets/${DATASET}/orb_images"
REPO_DIR="/home/adamb14/repos/ood-slam"
SPLITS_FILE="${REPO_DIR}/splits/${DATASET}_train.txt"
SPLITS_FILE_VAL="${REPO_DIR}/splits/${DATASET}_val.txt"

# --- STAGING LOOP ---
# We iterate through the split files to grab exactly what we need
cat ${SPLITS_FILE} ${SPLITS_FILE_VAL} | while read seq_path; do
    # seq_path looks like: "gascola/Hard/P004"
    
    # 1. Parse components
    domain=$(echo "$seq_path" | cut -d'/' -f1)
    diff=$(echo "$seq_path" | cut -d'/' -f2)
    seq_id=$(echo "$seq_path" | cut -d'/' -f3)
    
    # 2. Prepare Directory
    TARGET_DIR="${SLURM_TMPDIR}/${domain}/${diff}"
    mkdir -p "${TARGET_DIR}"
    
    # 3. Stage RGB (Zip contains P004 folder)
    # Source: tartanair_zipped/gascola/Hard/P004.zip
    rgb_zip="${DATA_SOURCE}/${domain}/${diff}/${seq_id}.zip"
    
    if [ -f "$rgb_zip" ]; then
        cp "$rgb_zip" "${TARGET_DIR}/"
        unzip -q "${TARGET_DIR}/${seq_id}.zip" -d "${TARGET_DIR}"
        rm "${TARGET_DIR}/${seq_id}.zip"
    fi
    
    # 4. Stage ORB (Tar contains P004 folder)
    # Source: tartanair/orb_images/gascola_Hard_P004_orb.tar.gz
    # Note: TartanAir ORB filenames usually flatten the path with underscores
    orb_tar="${ORB_SOURCE}/${domain}_${diff}_${seq_id}_orb.tar.gz"
    
    if [ -f "$orb_tar" ]; then
        cp "$orb_tar" "${TARGET_DIR}/"
        # Extracting inside 'gascola/Hard' creates 'gascola/Hard/P004/orb_semantic'
        tar -xzf "${TARGET_DIR}/$(basename $orb_tar)" -C "${TARGET_DIR}"
        rm "${TARGET_DIR}/$(basename $orb_tar)"
    fi
    
    # 5. Stage Labels
    mkdir -p "${SLURM_TMPDIR}/labels/${seq_path}"
    cp "${LABELS_SOURCE}/${seq_path}/labels.csv" "${SLURM_TMPDIR}/labels/${seq_path}/" 2>/dev/null
done

echo "Staging Complete. Structure check:"
ls -R ${SLURM_TMPDIR} | head -20

# --- RUN ---
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