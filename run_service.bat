@echo off
REM Activate the Conda environment
call C:\Users\nj.camera\Miniconda3\Scripts\activate.bat C:\Users\nj.camera\Miniconda3\envs\sentinel

REM Change to the project directory containing main.py
cd /d "C:\Users\nj.camera\indriya\ambient-machine"

REM Run the application
python main.py