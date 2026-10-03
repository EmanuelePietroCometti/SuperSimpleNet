#!/bin/bash
# Campagna HPO di SuperSimpleNet su Legion: in parallelo ma con lo stesso protocollo sequenziale di
# hyperparameter_finetuning.py (ogni studio Optuna resta in mano a un solo job alla volta).
#
#   ./submit_hpo.sh all     <categoria>   stadio A (un job per architettura) + stadio B (un job per architettura
#                                         del fronte; parte da solo quando tutto l'A e' finito)
#   ./submit_hpo.sh screen  <categoria>   solo stadio A
#   ./submit_hpo.sh tune    <categoria>   solo stadio B (lo stadio A deve essere gia' concluso)
#   ./submit_hpo.sh status  <categoria>   avanzamento per studio
#   ./submit_hpo.sh collect <categoria>   riunisce i DB e stampa i report finali
#
# Variabili opzionali (esportate o messe davanti al comando):
#   HPO_ARGS="--epochs 300 --n_trials 30 --tune_batch"   argomenti extra di hyperparameter_finetuning.py;
#                                                        DEVONO essere identici in tutti i comandi della campagna
#   SCREEN_CHAIN=2 TUNE_CHAIN=4   job accodati per unita': se uno muore per timeout riparte il successivo
#   TIME=48:00:00                 walltime per job (default: quello di hpo.sbatch)
#   PARTITION=gpu_a40_ext         partizione (default: gpu_a40, max 24h; la _ext arriva a 120h: con
#                                 TIME=120:00:00 bastano meno job accodati, es. TUNE_CHAIN=1)
#   SSN_ENV=$HOME/env_ssn         ambiente Python da usare (default: ~/env_ema_tesi); va esportato anche per hpo.sbatch
#   EXCLUDE=compute-4-13          nodi da escludere (es. uno che fallisce sempre con JobLaunchFailure)
#   SCREEN_ARCHS="resnet18-layer1_2 ..."   sottoinsieme dello stadio A (default: tutte e 12)
#   RUN_TAG=smoke                 campagna separata (altri DB, altre cartelle), per i test

MODE=$1
CAT=$2
if [ -z "$CAT" ]; then
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

cd "$(dirname "$(readlink -f "$0")")" || exit 1
export PATH="${SSN_ENV:-$HOME/env_ema_tesi}/bin:$PATH"
export HPO_ARGS RUN_TAG
mkdir -p logs

SCREEN_CHAIN=${SCREEN_CHAIN:-2}
TUNE_CHAIN=${TUNE_CHAIN:-4}
TAGGED=$CAT${RUN_TAG:+_$RUN_TAG}
MIRROR=$PWD/hpo_runs/$TAGGED
DIRS=(--state-dir "$MIRROR")
[ -n "$SCRATCH_FLASH" ] && DIRS=(--state-dir "${STATE_ROOT:-$SCRATCH_FLASH/ssn_hpo}/$TAGGED" --state-dir "$MIRROR")

# numero di architetture portate al tuning: come --top_k dello script (default 3)
TOP_K=$(sed -n 's/.*--top_k[ =]\{1,\}\([0-9]\{1,\}\).*/\1/p' <<< "$HPO_ARGS")
TOP_K=${TOP_K:-3}

submit() {  # submit <nome> <dipendenza|""> <argomenti di hpo.sbatch>  ->  stampa l'ID del job
    local name=$1 dep=$2 out
    shift 2
    out=$(sbatch --parsable --job-name="$name" ${TIME:+--time="$TIME"} ${PARTITION:+--partition="$PARTITION"} ${EXCLUDE:+--exclude="$EXCLUDE"} ${dep:+--dependency="$dep"} hpo.sbatch "$@") \
        || { echo "sbatch fallito per $name" >&2; exit 1; }
    echo "${out%%;*}"
}

chain() {  # chain <nome> <n> <dipendenza iniziale|""> <argomenti>  ->  stampa l'ID dell'ULTIMO job della catena
    local name=$1 n=$2 dep=$3 id i
    shift 3
    id=$(submit "$name" "$dep" "$@") || exit 1
    for ((i = 2; i <= n; i++)); do
        id=$(submit "$name" "afterany:$id" "$@") || exit 1  # parte comunque; se l'unita' e' gia' finita esce subito
    done
    echo "$id"
}

submit_screen() {
    local archs=${SCREEN_ARCHS:-$(for b in resnet18 resnet34 resnet50 wide_resnet50_2; do
        for l in layer1_2 layer2_3 layer1_2_3; do echo "$b-$l"; done; done)}
    local a id
    SCREEN_IDS=()
    for a in $archs; do
        id=$(chain "hpo_${TAGGED}_s_$a" "$SCREEN_CHAIN" "" screen "$CAT" "$a") || exit 1
        SCREEN_IDS+=("$id")
        echo "screen $a -> ultimo job della catena: $id"
    done
}

submit_tune() {  # $1 = dipendenza iniziale (vuota se lo stadio A e' gia' concluso)
    local r id
    for ((r = 0; r < TOP_K; r++)); do
        id=$(chain "hpo_${TAGGED}_t_r$r" "$TUNE_CHAIN" "$1" tune "$CAT" "$r") || exit 1
        echo "tune rango $r -> ultimo job della catena: $id"
    done
}

case "$MODE" in
    screen)
        submit_screen
        ;;
    tune)
        submit_tune ""
        ;;
    all)
        submit_screen
        # il tuning parte solo se TUTTI gli screening sono andati a buon fine (afterok): niente fronte su dati parziali
        submit_tune "afterok:$(IFS=:; echo "${SCREEN_IDS[*]}")"
        ;;
    status)
        python hpo_tools.py status "${DIRS[@]}"
        exit 0
        ;;
    collect)
        FINAL=$MIRROR/final.db
        # unione dei DB e report importano torch/anomalib: la guida vieta lavoro pesante sul login node,
        # quindi girano su un nodo CPU con una sessione srun breve (COLLECT_LOCAL=1 per farlo qui)
        RUN=()
        if [ -z "$COLLECT_LOCAL" ] && [ -z "$SLURM_JOB_ID" ]; then
            RUN=(srun --partition=cpu_sapphire --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=1:00:00)
        fi
        "${RUN[@]}" python hpo_tools.py collect "${DIRS[@]}" --out "$FINAL" || exit 1
        # shellcheck disable=SC2086
        "${RUN[@]}" python hyperparameter_finetuning.py --stage report --category "$CAT" --datasets_folder . \
            --storage "sqlite:///$FINAL" --results_dir "$MIRROR/report" $HPO_ARGS
        exit 0
        ;;
    *)
        echo "modo sconosciuto: $MODE (all|screen|tune|status|collect)"
        exit 1
        ;;
esac

echo
echo "Controllo:   squeue -u \$USER -o \"%.10i %.34j %.8T %.10M %R\""
echo "Avanzamento: ./submit_hpo.sh status $CAT"
echo "A fine campagna: ./submit_hpo.sh collect $CAT"
