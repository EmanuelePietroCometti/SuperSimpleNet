#!/usr/bin/env python
"""Supporto alla campagna HPO parallela di SuperSimpleNet su Legion.

Schema: ogni job (uno per architettura nello screening, uno per architettura del fronte nel tuning) scrive
SOLO sul proprio DB SQLite, quindi non ci sono scritture concorrenti. Il protocollo di
hyperparameter_finetuning.py (che qui NON viene modificato) resta quello sequenziale: ogni studio Optuna e'
portato avanti da un solo processo alla volta.

  init-tune  crea il DB di un job di tuning copiandoci i risultati di screening dei singoli DB e sceglie la
             sua architettura con la regola originale (select_front dello script, non riscritta qui)
  status     numero di trial per studio
  collect    unisce tutti i DB in uno, leggibile da `hyperparameter_finetuning.py --stage report`
"""
from __future__ import annotations

import argparse
import shlex
import sys
from collections import Counter
from pathlib import Path

import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)
# PRUNED nello screening = architettura che non entra nella VRAM simulata: e' un esito valido, non un buco
KEEP = (TrialState.COMPLETE, TrialState.PRUNED)


def url(path) -> str:
    return f"sqlite:///{Path(path).resolve()}"


def find_db(name: str, dirs) -> Path | None:
    """Primo file `name` trovato nelle cartelle, in ordine di priorita' (live su scratch, poi mirror in home)."""
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def db_names(dirs) -> list[str]:
    return sorted({p.name for d in dirs if Path(d).is_dir() for p in Path(d).glob("*.db")})


def load_hpo(category: str, hpo_args: str):
    """Importa lo script HPO e ne fa il parsing con gli stessi default (e gli stessi override) dei job."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hyperparameter_finetuning as H

    sys.argv = ["hyperparameter_finetuning.py", "--stage", "tune", "--category", category,
                "--datasets_folder", "-"] + shlex.split(hpo_args)
    return H, H.parse_args()


def merge_screens(H, args, dirs, dst_url: str) -> None:
    """Riunisce in un unico studio di screening i risultati dei DB `screen_<arch>.db` (uno per architettura)."""
    expected = list(args.screen_archs) if args.screen_archs else list(H.ALL_ARCHS)
    dst = optuna.create_study(study_name=H.screen_name(args), storage=dst_url, direction="maximize",
                              load_if_exists=True)
    have = {t.params.get("arch") for t in dst.trials}
    missing = []
    for arch in expected:
        if arch in have:
            continue
        src = find_db(f"screen_{arch}.db", dirs)
        trials = []
        if src is not None:
            try:
                s = optuna.load_study(study_name=H.screen_name(args), storage=url(src))
                trials = [t for t in s.trials if t.state in KEEP and t.params.get("arch") == arch]
            except KeyError:
                pass
        if not trials:
            missing.append(arch)
            continue
        for t in trials:
            dst.add_trial(t)
    if missing:
        raise SystemExit("Screening incompleto, mancano: " + ", ".join(missing))


def cmd_init_tune(a) -> None:
    out = Path(a.out)
    H, args = load_hpo(a.category, a.hpo_args)
    args.storage = url(out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            merge_screens(H, args, a.state_dir, args.storage)
        except BaseException:
            out.unlink(missing_ok=True)  # niente DB a meta': il prossimo tentativo riparte pulito
            raise
    archs, front, best = H.select_front(args)
    print(f"Fronte di Pareto: {[f[0] for f in front]} | portate al tuning: {archs}", file=sys.stderr)
    Path(a.result_file).write_text(archs[a.rank] if a.rank < len(archs) else "")


def cmd_status(a) -> None:
    for name in db_names(a.state_dir):
        db = find_db(name, a.state_dir)
        try:
            for sn in optuna.get_all_study_names(url(db)):
                if name.startswith("tune_") and sn.endswith("_screen"):
                    continue  # copia dello screening dentro il DB di tuning
                s = optuna.load_study(study_name=sn, storage=url(db))
                trials = [t for t in s.trials if not t.user_attrs.get("from_screening")]
                c = Counter(t.state.name for t in trials)
                best = max((t.value for t in trials if t.state == TrialState.COMPLETE), default=None)
                counts = " ".join(f"{k}={v}" for k, v in sorted(c.items()))
                print(f"{name[:-3]:<42} {sn:<44} {counts:<26} best AUROC: {'-' if best is None else f'{best:.4f}'}")
        except Exception as e:  # DB momentaneamente bloccato da un job in scrittura, ecc.
            print(f"{name}: lettura non riuscita ({type(e).__name__}: {e})")


def cmd_collect(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    dst_url = url(out)
    for name in db_names(a.state_dir):
        db = find_db(name, a.state_dir)
        if name.startswith("screen_"):
            for sn in optuna.get_all_study_names(url(db)):
                src = optuna.load_study(study_name=sn, storage=url(db))
                dst = optuna.create_study(study_name=sn, storage=dst_url, direction="maximize", load_if_exists=True)
                for t in src.trials:
                    if t.state in KEEP:
                        dst.add_trial(t)
        elif name.startswith("tune_"):
            for sn in optuna.get_all_study_names(url(db)):
                if not sn.endswith("_screen"):  # lo screening arriva dai DB screen_*
                    optuna.copy_study(from_study_name=sn, from_storage=url(db), to_storage=dst_url)
    print(f"DB riunito: {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-tune")
    s.add_argument("--category", required=True)
    s.add_argument("--rank", type=int, required=True, help="posto (0,1,..) nell'elenco delle architetture scelte")
    s.add_argument("--out", required=True, help="DB del job di tuning")
    s.add_argument("--result-file", required=True, help="file dove scrivere l'architettura scelta (vuoto = nessuna)")
    s.add_argument("--state-dir", action="append", required=True, help="dove cercare i DB screen_*, in ordine di priorita'")
    s.add_argument("--hpo-args", default="", help="argomenti extra di hyperparameter_finetuning.py (stringa unica)")
    s.set_defaults(fn=cmd_init_tune)

    s = sub.add_parser("status")
    s.add_argument("--state-dir", action="append", required=True)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("collect")
    s.add_argument("--state-dir", action="append", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_collect)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
