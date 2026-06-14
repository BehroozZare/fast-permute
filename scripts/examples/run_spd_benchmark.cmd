@echo off
REM ------------------------------------------------------------------
REM Loop matrix_example over every .mtx file in input\ and write a
REM per-matrix JSON result to results\<solver>\<name>.json.
REM
REM Usage:
REM     scripts\examples\run_spd_benchmark.cmd                (default solver: cudss)
REM     scripts\examples\run_spd_benchmark.cmd mkl
REM     scripts\examples\run_spd_benchmark.cmd cholmod
REM     scripts\examples\run_spd_benchmark.cmd cudss 1024   (override patch size)
REM ------------------------------------------------------------------

setlocal EnableExtensions

set "SOLVER=%~1"
if "%SOLVER%"=="" set "SOLVER=cudss"

set "PATCH_SIZE=%~2"
if "%PATCH_SIZE%"=="" set "PATCH_SIZE=512"

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
set "INPUT_DIR=%REPO_ROOT%\input"
set "BINARY=%REPO_ROOT%\build\bin\Release\matrix_example.exe"
set "OUT_DIR=results\%SOLVER%"

if not exist "%BINARY%" (
    echo ERROR: matrix_example.exe not found at "%BINARY%"
    echo Build it first with:
    echo   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ^
        -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CHOLMOD=ON ^
        -DHOMA_WITH_MKL=ON -DHOMA_WITH_CUDSS=ON
    echo   cmake --build build --config Release
    exit /b 1
)

if not exist "%INPUT_DIR%\*.mtx" (
    echo ERROR: no .mtx files found in "%INPUT_DIR%"
    echo Run scripts\download_suitesparse_spd.py first to populate it.
    exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set /a COUNT=0

for %%f in ("%INPUT_DIR%\*.mtx") do (
    set /a COUNT+=1
    echo.
    echo ============================================================
    echo   Matrix: %%~nxf  ^|  solver: %SOLVER%  ^|  patch: %PATCH_SIZE%
    echo ============================================================
    "%BINARY%" -i "%%f" -s %SOLVER% -p %PATCH_SIZE% --out "%OUT_DIR%\%%~nf.json"    
)

echo.
echo ============================================================
echo Done. Processed %COUNT% matrices
echo Results written to "%OUT_DIR%"
echo ============================================================

endlocal
