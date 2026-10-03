@echo off
echo Installing ReprocessData service using NSSM...
set NSSM_EXE="C:\Users\nj.camera\indriya\ambient-machine\bin\nssm-2.24\win64\nssm.exe"

REM CHANGED: AmbientMachine -> ReprocessData  (all 5 lines below)
REM CHANGED: run_service.bat -> run_reprocess.bat
%NSSM_EXE% install ReprocessData "C:\Users\nj.camera\indriya\ambient-machine\run-reprocess.bat"
%NSSM_EXE% set ReprocessData AppDirectory "C:\Users\nj.camera\indriya\ambient-machine"
%NSSM_EXE% set ReprocessData AppStdout "C:\Users\nj.camera\indriya\ambient-machine\logs\reprocess.log"
%NSSM_EXE% set ReprocessData AppStderr "C:\Users\nj.camera\indriya\ambient-machine\logs\reprocess-error.log"

echo Starting the service...
%NSSM_EXE% start ReprocessData

echo Done! The ReprocessData service should now be running in the background.
pause