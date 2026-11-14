@echo off
rem Opens the live Isaac Sim view: SSH tunnel to the Verda GPU machine + TigerVNC viewer.
rem Close this window (or the viewer) to disconnect. Change SIM_IP if the instance is recreated.
set SIM_IP=86.38.238.226
echo Opening tunnel to %SIM_IP% ...
start "Isaac Sim tunnel (keep open)" ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 5901:127.0.0.1:5901 root@%SIM_IP%
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\TigerVNC\vncviewer.exe" localhost:5901
