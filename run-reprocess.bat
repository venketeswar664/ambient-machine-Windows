@echo off
REM Force UTF-8 so the emoji in database.py cannot crash print() when
REM output is redirected to a log file (this was the exit-code-1 cause).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Activate the Conda environment
call C:\Users\nj.camera\Miniconda3\Scripts\activate.bat C:\Users\nj.camera\Miniconda3\envs\sentinel

REM Change to the project directory
cd /d "C:\Users\nj.camera\indriya\ambient-machine"

:loop
REM reprocess_date.py REQUIRES a positional date argument - without it
REM argparse exits with code 2 immediately.
python reprocess_date.py 2026-09-25 --native-width 1280 
