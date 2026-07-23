#!/usr/bin/env python3
"""
check_contract_sync.py — Verifica che export_common.py sia IDENTICO nelle tre
repo e non manomesso.

Il contratto di export (export_common.py) e' l'unica fonte di verita' condivisa
dalle 4 architetture. Se una copia diverge senza che le altre siano aggiornate,
i grafi esportati smettono silenziosamente di essere confrontabili. Questo
script rende la divergenza un errore rumoroso (exit code 1), adatto a un
pre-commit hook o alla CI.

Due controlli:
  1. HASH DI INTEGRITA': in ogni copia, sha256(contenuto con EXPORT_COMMON_SHA256
     azzerato) deve coincidere con il valore di EXPORT_COMMON_SHA256 dichiarato.
     Intercetta una modifica al file non accompagnata dall'aggiornamento
     dell'hash.
  2. IDENTITA' BYTE-PER-BYTE: le tre copie devono essere identiche.

Uso:
    python check_contract_sync.py              # verifica (exit 1 se diverge)
    python check_contract_sync.py --write-hash # ricalcola e scrive l'hash in
                                               # TUTTE le copie (dopo una modifica)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# Le tre repo sono sorelle sotto la stessa cartella genitore di questo script
# (SuperSimpleNet/). Risolviamo i percorsi da li' invece di hardcodarli assoluti.
_REPO_ROOT = Path(__file__).resolve().parent          # .../SuperSimpleNet
_PARENT = _REPO_ROOT.parent                            # .../UNI

COPIES = [
    _REPO_ROOT / "export_common.py",
    _PARENT / "anomaly_detection_for_textile_industry" / "src" / "export_common.py",
    _PARENT / "SK-RD4AD" / "export_common.py",
]

_HASH_LINE_RE = re.compile(
    r'^(EXPORT_COMMON_SHA256\s*=\s*")([0-9a-fA-F]{64})(")', re.MULTILINE
)
_ZERO = "0" * 64


def _canonical_bytes(text: str) -> bytes:
    """Testo con il valore dell'hash azzerato, per calcolare l'hash del
    contenuto in modo indipendente dall'hash stesso (auto-referenza)."""
    canon, n = _HASH_LINE_RE.subn(rf"\g<1>{_ZERO}\g<3>", text)
    if n != 1:
        raise ValueError(
            "EXPORT_COMMON_SHA256 = \"<64 hex>\" non trovato (o presente piu' "
            "volte): il file non ha la riga di hash attesa."
        )
    return canon.encode("utf-8")


def _declared_hash(text: str) -> str:
    m = _HASH_LINE_RE.search(text)
    if not m:
        raise ValueError("EXPORT_COMMON_SHA256 non trovato.")
    return m.group(2).lower()


def _compute_hash(text: str) -> str:
    return hashlib.sha256(_canonical_bytes(text)).hexdigest()


def write_hash() -> int:
    """Ricalcola l'hash dalla copia primaria (SuperSimpleNet) e lo scrive,
    identico, in tutte le copie. Non tocca il resto del contenuto."""
    primary = COPIES[0]
    if not primary.exists():
        print(f"[ERRORE] copia primaria mancante: {primary}", file=sys.stderr)
        return 1
    text = primary.read_text(encoding="utf-8")
    new_hash = _compute_hash(text)
    for path in COPIES:
        if not path.exists():
            print(f"[ERRORE] copia mancante: {path}", file=sys.stderr)
            return 1
        t = path.read_text(encoding="utf-8")
        t2 = _HASH_LINE_RE.sub(rf"\g<1>{new_hash}\g<3>", t, count=1)
        path.write_text(t2, encoding="utf-8")
        print(f"[OK] hash scritto in {path}")
    print(f"\nEXPORT_COMMON_SHA256 = {new_hash}")
    return 0


def check() -> int:
    missing = [p for p in COPIES if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERRORE] copia mancante: {p}", file=sys.stderr)
        return 1

    texts = {p: p.read_text(encoding="utf-8") for p in COPIES}
    ok = True

    # 1. hash di integrita' per copia
    for p, t in texts.items():
        try:
            declared, computed = _declared_hash(t), _compute_hash(t)
        except ValueError as e:
            print(f"[ERRORE] {p}: {e}", file=sys.stderr)
            ok = False
            continue
        if declared != computed:
            print(f"[ERRORE] hash non corrispondente in {p}\n"
                  f"         dichiarato: {declared}\n"
                  f"         calcolato : {computed}\n"
                  "         → il file e' stato modificato senza aggiornare "
                  "l'hash. Esegui: python check_contract_sync.py --write-hash",
                  file=sys.stderr)
            ok = False

    # 2. identita' byte-per-byte tra le copie
    raw = {p: p.read_bytes() for p in COPIES}
    reference = raw[COPIES[0]]
    for p in COPIES[1:]:
        if raw[p] != reference:
            print(f"[ERRORE] {p} diverge dalla copia primaria {COPIES[0]}.\n"
                  "         Le tre copie di export_common.py devono essere "
                  "identiche byte per byte. Propaga la versione corretta e "
                  "riesegui --write-hash.", file=sys.stderr)
            ok = False

    if ok:
        print("[OK] export_common.py identico e integro nelle tre repo:")
        for p in COPIES:
            print(f"     - {p}")
        print(f"     sha256(contenuto) = {_declared_hash(texts[COPIES[0]])}")
        return 0
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-hash", action="store_true",
                    help="Ricalcola l'hash e scrivilo in tutte le copie.")
    args = ap.parse_args()
    return write_hash() if args.write_hash else check()


if __name__ == "__main__":
    sys.exit(main())
