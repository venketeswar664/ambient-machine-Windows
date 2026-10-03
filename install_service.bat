@echo off
echo Installing AmbientMachine service using NSSM...
set NSSM_EXE="C:\Users\nj.camera\indriya\ambient-machine\bin\nssm-2.24\win64\nssm.exe"

%NSSM_EXE% install AmbientMachine "C:\Users\nj.camera\indriya\ambient-machine\run_service.bat"
%NSSM_EXE% set AmbientMachine AppDirectory "C:\Users\nj.camera\indriya\ambient-machine"
%NSSM_EXE% set AmbientMachine AppStdout "C:\Users\nj.camera\indriya\ambient-machine\logs\service.log"
%NSSM_EXE% set AmbientMachine AppStderr "C:\Users\nj.camera\indriya\ambient-machine\logs\service-error.log"

echo Starting the service...
%NSSM_EXE% start AmbientMachine

echo Done! The AmbientMachine service should now be running in the background.
pause