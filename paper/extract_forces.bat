@echo off
setlocal

set "LOGDIR=0"

if exist forcesX.txt del forcesX.txt
if exist forcesY.txt del forcesY.txt
if exist forcesZ.txt del forcesZ.txt

powershell -NoProfile -Command ^
    "$logdir = '%LOGDIR%';" ^
    "$xAll = @(); $yAll = @(); $zAll = @();" ^
    "Get-ChildItem \"$logdir\\*.log\" | ForEach-Object {" ^
    "  $inForces = $false; $data = $false;" ^
    "  Get-Content $_.FullName | ForEach-Object {" ^
    "    if ($_ -match 'Forces \(Hartrees/Bohr\)') { $inForces = $true; $data = $false }" ^
    "    elseif ($inForces -and $_ -match '^ -{10,}') {" ^
    "      if (-not $data) { $data = $true } else { $inForces = $false; $data = $false }" ^
    "    }" ^
    "    elseif ($data -and $_ -match '^\s+\d+\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)') {" ^
    "      $xAll += $matches[1]; $yAll += $matches[2]; $zAll += $matches[3]" ^
    "    }" ^
    "  }" ^
    "};" ^
    "$xAll | Out-File -Encoding ASCII forcesX.txt;" ^
    "$yAll | Out-File -Encoding ASCII forcesY.txt;" ^
    "$zAll | Out-File -Encoding ASCII forcesZ.txt"

echo Done. Forces written to forcesX.txt, forcesY.txt, forcesZ.txt
