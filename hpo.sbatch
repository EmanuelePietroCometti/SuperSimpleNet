#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s346291@studenti.polito.it
#SBATCH --output=logs/%x_%j.out

# Worker della campagna HPO: un job = una unita' di lavoro. Si lancia con submit_hpo.sh, non a mano.
#   sbatch hpo.sbatch screen <CATEGORIA> <ARCH>   stadio A, una architettura (es. resnet18-layer1_2)
#   sbatch hpo.sbatch tune   <CATEGORIA> <RANK>   stadio B, architettura di posto RANK (0,1,..) del fronte
# Se il job muore per timeout riparte dal DB Optuna al rilancio (submit_hpo.sh accoda gia' i rilanci).
# Variabili ereditate dall'ambiente: HPO_ARGS, RUN_TAG, DATA_HOME, STATE_ROOT.

ROLE=$1
CAT=$2
UNIT=$3
case "$ROLE" in
    screen|tune) ;;
    *) echo "uso: sbatch hpo.sbatch screen|tune <categoria> <arch|rank>"; exit 2 ;;
esac

REPO=$SLURM_SUBMIT_DIR
cd "$REPO" || exit 2

HPO_SCRIPT=hyperparameter_finetuning.py
DATA_HOME=${DATA_HOME:-/home/mla_group_08/TESI_EMA/datasets}
if [ -z "$STATE_ROOT" ]; then
    [ -n "$SCRATCH_FLASH" ] || { echo "SCRATCH_FLASH non definita"; exit 2; }
    STATE_ROOT=$SCRATCH_FLASH/ssn_hpo
fi
TAGGED=$CAT${RUN_TAG:+_$RUN_TAG}
NAME=${ROLE}_${UNIT}
LIVE=$STATE_ROOT/$TAGGED            # DB "vivo" (su scratch): lo scrive solo questo job
MIRROR=$REPO/hpo_runs/$TAGGED       # in home: istantanee del DB, marker .done, risultati
DB=$LIVE/$NAME.db
WORK=${SCRATCH_FLASH:-${TMPDIR:-/tmp}}/ssn_hpo_work/${TAGGED}_${NAME}_$SLURM_JOB_ID
mkdir -p "$LIVE" "$MIRROR" "$WORK/datasets" "$WORK/results" || {
    echo "[$NAME] ERRORE: non riesco a creare cartelle su scratch ($LIVE, $WORK) sul nodo $(hostname)."
    echo "[$NAME] SCRATCH_FLASH='$SCRATCH_FLASH': non esiste o non e' scrivibile da questo nodo."
    exit 5
}

export PATH="${SSN_ENV:-$HOME/env_ema_tesi}/bin:$PATH"
IFS=' ' read -r -a EXTRA <<< "$HPO_ARGS"

echo "[$NAME] categoria=$TAGGED job=$SLURM_JOB_ID nodo=$(hostname) $(date)"
echo "[$NAME] HPO_ARGS='$HPO_ARGS'"

if [ -f "$MIRROR/$NAME.done" ]; then
    echo "[$NAME] gia' completato, niente da fare"
    rmdir "$WORK/datasets" "$WORK/results" "$WORK" 2>/dev/null
    exit 0
fi

# copia consistente del DB live nel mirror (API backup di sqlite, sicura anche con un job che scrive)
snapshot() {
    [ -f "$DB" ] || return 0
    python - "$DB" "$MIRROR/$NAME.db" <<'EOF' || echo "[snapshot] non riuscito (ignorato)"
import os, sqlite3, sys
src, dst = sys.argv[1:3]
a = sqlite3.connect(src, timeout=60)
b = sqlite3.connect(dst + ".tmp")
a.backup(b)
b.close()
a.close()
os.replace(dst + ".tmp", dst)
EOF
}

SNAP_PID=
stop_snapshots() {
    if [ -n "$SNAP_PID" ]; then
        kill "$SNAP_PID" 2>/dev/null
        wait "$SNAP_PID" 2>/dev/null
        SNAP_PID=
    fi
}
cleanup() {
    stop_snapshots
    rm -rf "$WORK"
}
trap cleanup EXIT

# scratch svuotato o job ripartito su un altro nodo: si riparte dall'ultima istantanea in home
if [ ! -f "$DB" ] && [ -f "$MIRROR/$NAME.db" ]; then
    cp "$MIRROR/$NAME.db" "$DB" && echo "[$NAME] DB ripristinato dall'istantanea in home"
fi

# lo script HPO, se non trova la GPU, va avanti su CPU senza fermarsi: qui ci si ferma subito
python - <<'EOF' || exit 2
import sys
import optuna
import torch
print(f"python {sys.version.split()[0]} | torch {torch.__version__} (cuda {torch.version.cuda}) | optuna {optuna.__version__}")
if not torch.cuda.is_available():
    sys.exit("ERRORE: CUDA non disponibile, interrompo (l'HPO girerebbe su CPU)")
print("GPU:", torch.cuda.get_device_name(0))
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv

# la sola categoria necessaria, copiata su scratch
[ -d "$DATA_HOME/$CAT" ] || { echo "[$NAME] dataset non trovato in $DATA_HOME/$CAT"; exit 3; }
cp -r "$DATA_HOME/$CAT" "$WORK/datasets/" || { echo "[$NAME] copia del dataset in $WORK/datasets fallita"; exit 3; }

if [ "$ROLE" = screen ]; then
    ARGS=(--stage screen --screen_archs "$UNIT")
else
    # unisce i DB di screening nel DB di questo job e sceglie l'architettura col fronte di Pareto originale
    python -u hpo_tools.py init-tune --category "$CAT" --rank "$UNIT" --out "$DB" \
        --state-dir "$LIVE" --state-dir "$MIRROR" --hpo-args="$HPO_ARGS" --result-file "$WORK/arch.txt" \
        || { echo "[$NAME] init-tune fallito"; exit 4; }
    ARCH=$(cat "$WORK/arch.txt")
    if [ -z "$ARCH" ]; then
        echo "[$NAME] il fronte ha meno architetture del rango $UNIT: niente da fare"
        touch "$MIRROR/$NAME.done"
        exit 0
    fi
    echo "[$NAME] architettura: $ARCH"
    ARGS=(--stage tune --archs "$ARCH")
fi

# istantanea in home ogni 10 minuti; il trap uccide anche lo sleep in corso, altrimenti resterebbe orfano
(
    trap 'kill "$SLP" 2>/dev/null; exit 0' TERM
    while true; do
        sleep 600 &
        SLP=$!
        wait "$SLP"
        snapshot
    done
) &
SNAP_PID=$!

# gli argomenti dell'utente vanno PRIMA: stage/storage/categoria/cartelle qui sotto non si possono sovrascrivere
python -u "$HPO_SCRIPT" "${EXTRA[@]}" "${ARGS[@]}" \
    --category "$CAT" \
    --datasets_folder "$WORK/datasets" \
    --storage "sqlite:///$DB" \
    --results_dir "$WORK/results"
STATUS=$?

stop_snapshots
snapshot
mkdir -p "$MIRROR/results_$NAME"
rsync -a "$WORK/results/" "$MIRROR/results_$NAME/"

echo "[$NAME] terminato con status $STATUS $(date)"
[ $STATUS -eq 0 ] && touch "$MIRROR/$NAME.done"
exit $STATUS
