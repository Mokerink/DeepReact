@echo off
setlocal enabledelayedexpansion

set "LOGDIR=0"
set "OUTFILE=energies.txt"

if exist "%OUTFILE%" del "%OUTFILE%"

for %%f in ("%LOGDIR%\*.log") do (
    for /f "tokens=4 delims== " %%e in ('findstr /C:"SCF Done:" "%%f" 2^>nul') do (
        echo %%e>> "%OUTFILE%"
    )
)

echo Done. Energies written to %OUTFILE%
