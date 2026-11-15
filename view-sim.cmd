@echo off
rem Opens the live Isaac Sim view: SSH tunnel to the GPU machine + TigerVNC viewer.
rem Close this window (or the viewer) to disconnect.
rem The server address is read from %SIM_IP%, or from sim-ip.txt next to this file (git-ignored, one line).
if "%SIM_IP%"=="" if exist "%~dp0sim-ip.txt" set /p SIM_IP=<"%~dp0sim-ip.txt"
if "%SIM_IP%"=="" (
  echo Set SIM_IP or put the server address in sim-ip.txt next to this file.
  pause
  exit /b 1
)
echo Opening tunnel to %SIM_IP% ...
start "Isaac Sim tunnel (keep open)" ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 5901:127.0.0.1:5901 root@%SIM_IP%
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\TigerVNC\vncviewer.exe" localhost:5901
