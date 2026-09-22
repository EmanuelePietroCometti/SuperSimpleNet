@echo off
setlocal

REM ==========================================================
REM  SuperSimpleNet - configurazione standard
REM  Output: results\ssn_<classe>\seed_<seed>\...
REM ==========================================================

REM Classi e seed
set "classes=carpet reda_baseline reda_dustOnValidation reda_dustOnValidationAndTrain"
set "seeds=0 1 2 42 101"

REM Paradigma: sup (mixed supervision, default di train.py) oppure unsup
set "MODE=sup"

set "DATA_ROOT=mvtec"
set "RESULTS_ROOT=results"

for %%c in (%classes%) do (
    for %%s in (%seeds%) do (
        echo ==========================================================
        echo Starting run -^> Class: %%c ^| Seed: %%s ^| Project: ssn_%%c
        echo ==========================================================

        python train.py ^
            --mode %MODE% ^
            --dataset mvtec ^
            --category %%c ^
            --datasets_folder %DATA_ROOT% ^
            --results_save_path %RESULTS_ROOT%\ssn_%%c ^
            --setup_name seed_%%s ^
            --seed %%s ^
            --backbone wide_resnet50_2 ^
            --layers layer2 layer3 ^
            --image_size 256 256 ^
            --perlin_thr 0.2 ^
            --noise_std 0.015 ^
            --epochs 100 ^
            --batch 4 ^
            --seg_lr 0.0002 ^
            --dec_lr 0.0002 ^
            --adapt_lr 0.0001 ^
            --patch_size 3 ^
            --gamma 0.4 ^
            --eval_step_size 5 ^
            --th 0.5 ^
            --num_workers 4

        if errorlevel 1 (
            echo [ERRORE] Run fallita -^> Class: %%c ^| Seed: %%s
        ) else (
            echo Finished run -^> Class: %%c ^| Seed: %%s
        )
        echo.
    )
)

echo Tutti gli esperimenti sono stati completati!
pause