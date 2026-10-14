@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM  SuperSimpleNet - multi-seed della miglior config HPO (trial 30, AUROC 0.9942)
REM  Output: results\ssn_<classe>\seed_<seed>\...
REM  Dopo ogni training parte eval.py sul checkpoint
REM ==========================================================

REM Classi e seed (42 = seed dell'HPO: serve da controllo di riproducibilita')
set "classes=tessuto_nero_dust_train"
set "seeds=0 1 2 42 101"

REM Paradigma: sup (mixed supervision)
set "MODE=sup"

REM Architettura
set "BACKBONE=resnet18"
set "LAYERS=layer1 layer2"

REM Iperparametri del trial 30 (lr_scale = 0.125 su 2e-4 / 2e-4 / 1e-4)
set "BATCH=8"
set "SEG_LR=2.5e-5"
set "DEC_LR=2.5e-5"
set "ADAPT_LR=1.25e-5"
set "GAMMA=0.8"
set "NOISE_STD=0.0075"
set "PERLIN_THR=0.6"
set "PATCH_SIZE=5"

REM Come nell'HPO: 300 epoche, valutazione ogni 20 (decide quali checkpoint si possono scegliere)
set "EPOCHS=300"
set "EVAL_STEP=20"

set "DATA_ROOT=D:\emanuele\Code\dataset"
set "RESULTS_ROOT=results"

for %%c in (%classes%) do (
    for %%s in (%seeds%) do (
        echo ==========================================================
        echo Starting run -^> Class: %%c ^| Seed: %%s ^| Mode: %MODE%
        echo ==========================================================

        python train.py ^
            --mode %MODE% ^
            --dataset mvtec ^
            --category %%c ^
            --datasets_folder %DATA_ROOT% ^
            --results_save_path %RESULTS_ROOT%\ssn_%%c ^
            --setup_name seed_%%s ^
            --seed %%s ^
            --backbone %BACKBONE% ^
            --layers %LAYERS% ^
            --image_size 256 256 ^
            --perlin_thr %PERLIN_THR% ^
            --noise_std %NOISE_STD% ^
            --epochs %EPOCHS% ^
            --batch %BATCH% ^
            --seg_lr %SEG_LR% ^
            --dec_lr %DEC_LR% ^
            --adapt_lr %ADAPT_LR% ^
            --patch_size %PATCH_SIZE% ^
            --gamma %GAMMA% ^
            --eval_step_size %EVAL_STEP% ^
            --th 0.5 ^
            --num_workers 1

        if errorlevel 1 (
            echo [ERRORE] Run fallita -^> Class: %%c ^| Seed: %%s
        ) else (
            echo Finished run -^> Class: %%c ^| Seed: %%s

            set "CKPT=%RESULTS_ROOT%\ssn_%%c\seed_%%s\checkpoints\mvtec\%%c\1\weights.pt"
            if exist "!CKPT!" (
                echo Starting eval -^> Class: %%c ^| Seed: %%s
                python eval.py ^
                    "!CKPT!" ^
                    --dataset mvtec ^
                    --category %%c ^
                    --datasets_folder %DATA_ROOT% ^
                    --results_save_path %RESULTS_ROOT%\ssn_%%c\seed_%%s ^
                    --image_size 256 256 ^
                    --backbone %BACKBONE% ^
                    --layers %LAYERS% ^
                    --patch_size %PATCH_SIZE% ^
                    --batch %BATCH% ^
                    --num_workers 1 ^
                    --seed %%s

                if errorlevel 1 (
                    echo [ERRORE] Eval fallita -^> Class: %%c ^| Seed: %%s
                ) else (
                    echo Finished eval -^> Class: %%c ^| Seed: %%s
                )
            ) else (
                echo [ATTENZIONE] Checkpoint non trovato -^> !CKPT!
            )
        )
        echo.
    )
)

echo Tutti gli esperimenti sono stati completati!
pause