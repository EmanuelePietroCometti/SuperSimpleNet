#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s346291@studenti.polito.it
#SBATCH --output=logs/%x_%j.out

# uso: sbatch run_experiment.sbatch <CLASS> [SEED ...]
# un job per dataset: train + eval dei vari seed vengono eseguiti in sequenza
CLASS=$1
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 42 101)

REPO=$SLURM_SUBMIT_DIR
DATA_HOME=/home/mla_group_08/TESI_EMA/datasets
WORK=$SCRATCH_FLASH/ssn_${CLASS}/$SLURM_JOB_ID

# paradigma: sup (mixed supervision) oppure unsup
MODE=sup
[ "$CLASS" = "carpet" ] && MODE=unsup

# parametri condivisi tra train ed eval
IMAGE_SIZE="256 256"
BACKBONE=wide_resnet50_2
LAYERS="layer2 layer3"
PATCH_SIZE=3
BATCH=32
NUM_WORKERS=4

export PATH="$HOME/env_ema_tesi/bin:$PATH"
echo "Class: $CLASS | Mode: $MODE | Seeds: ${SEEDS[*]} | Node: $(hostname) | Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# copia la sola classe necessaria su scratch, una volta per tutte le run
mkdir -p "$WORK/datasets" "$WORK/runs"
cp -r "$DATA_HOME/$CLASS" "$WORK/datasets/"

cd "$REPO"

FAILED=()
for SEED in "${SEEDS[@]}"; do
    SETUP=seed_${SEED}
    RUN_WORK=$WORK/runs/$SETUP
    DEST=$REPO/results/ssn_${CLASS}/$SETUP
    WEIGHTS=$RUN_WORK/checkpoints/mvtec/$CLASS/1/weights.pt

    echo "=========================================================="
    echo "Starting run -> Class: $CLASS | Seed: $SEED | Mode: $MODE | $(date)"
    echo "=========================================================="

    # training: legge e scrive solo su scratch
    python -u "$REPO/train.py" \
        --mode "$MODE" \
        --dataset mvtec \
        --category "$CLASS" \
        --datasets_folder "$WORK/datasets" \
        --results_save_path "$WORK/runs" \
        --setup_name "$SETUP" \
        --seed "$SEED" \
        --backbone "$BACKBONE" \
        --layers $LAYERS \
        --image_size $IMAGE_SIZE \
        --perlin_thr 0.2 \
        --noise_std 0.015 \
        --epochs 300 \
        --batch "$BATCH" \
        --seg_lr 0.0002 \
        --dec_lr 0.0002 \
        --adapt_lr 0.0001 \
        --patch_size "$PATCH_SIZE" \
        --gamma 0.4 \
        --eval_step_size 20 \
        --th 0.5 \
        --num_workers "$NUM_WORKERS"
    STATUS=$?

    # eval sul checkpoint appena salvato (saltata se il train è fallito)
    if [ $STATUS -eq 0 ] && [ -f "$WEIGHTS" ]; then
        echo "---------- Eval -> Class: $CLASS | Seed: $SEED | $(date) ----------"
        python -u "$REPO/eval.py" "$WEIGHTS" \
            --dataset mvtec \
            --category "$CLASS" \
            --datasets_folder "$WORK/datasets" \
            --results_save_path "$RUN_WORK/eval" \
            --image_size $IMAGE_SIZE \
            --batch "$BATCH" \
            --num_workers "$NUM_WORKERS" \
            --backbone "$BACKBONE" \
            --layers $LAYERS \
            --patch_size "$PATCH_SIZE" \
            --seed "$SEED"
        EVAL_STATUS=$?
        [ $EVAL_STATUS -ne 0 ] && FAILED+=("eval:$SEED")
    else
        echo "[ERRORE] Train fallito (status $STATUS) -> eval saltata"
        FAILED+=("train:$SEED")
        EVAL_STATUS=skipped
    fi

    # riporta in home la run appena conclusa, così non si perde se il job va in timeout
    mkdir -p "$DEST"
    rsync -a "$RUN_WORK/" "$DEST/" && rm -rf "$RUN_WORK"

    echo "Finished run -> Class: $CLASS | Seed: $SEED | train: $STATUS | eval: $EVAL_STATUS | $(date)"
done

# pulisce scratch (dataset incluso)
cd "$REPO"
rm -rf "$WORK"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Run fallite: ${FAILED[*]}"
    exit 1
fi
echo "Tutte le run per $CLASS completate"
