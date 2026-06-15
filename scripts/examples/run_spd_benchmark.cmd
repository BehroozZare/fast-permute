@echo off
REM ------------------------------------------------------------------
REM Loop matrix_example over every .mtx file recursively under input\
REM and write a per-matrix JSON result to
REM results\<solver>\<relpath>\<name>_<precision>.json, mirroring the input
REM directory layout (so groups like input\HB\foo.mtx end up in
REM results\<solver>\HB\foo_double.json and foo_float.json).
REM
REM Usage:
REM     scripts\examples\run_spd_benchmark.cmd                (default solver: cudss)
REM     scripts\examples\run_spd_benchmark.cmd mkl
REM     scripts\examples\run_spd_benchmark.cmd cholmod
REM     scripts\examples\run_spd_benchmark.cmd cudss 1024     (override patch size)
REM ------------------------------------------------------------------

setlocal EnableExtensions EnableDelayedExpansion

set "SOLVER=%~1"
if "%SOLVER%"=="" set "SOLVER=cudss"

set "PATCH_SIZE=%~2"
if "%PATCH_SIZE%"=="" set "PATCH_SIZE=512"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul
set "REPO_ROOT=%CD%"
popd >nul

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

if not exist "%INPUT_DIR%" (
    echo ERROR: input directory not found: "%INPUT_DIR%"
    exit /b 1
)

dir /b /s /a-d "%INPUT_DIR%\*.mtx" >nul 2>&1
if errorlevel 1 (
    echo ERROR: no .mtx files found under "%INPUT_DIR%"
    echo Run scripts\download_suitesparse_spd.py first to populate it.
    exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set /a COUNT=0

for /R "%INPUT_DIR%" %%f in (*.mtx) do (
    set /a COUNT+=1
    set "FULL_DIR=%%~dpf"
    set "REL_DIR=!FULL_DIR:%INPUT_DIR%\=!"
    if not "!REL_DIR!"=="" (
        if not exist "%OUT_DIR%\!REL_DIR!" mkdir "%OUT_DIR%\!REL_DIR!" 2>nul
    )
    echo.
    echo ============================================================
    echo   Matrix: !REL_DIR!%%~nxf  ^|  solver: %SOLVER%  ^|  patch: %PATCH_SIZE%
    echo ============================================================
    "%BINARY%" -i "%%f" -s %SOLVER% -p %PATCH_SIZE% --precision double --make-spd-from-pattern --out "%OUT_DIR%\!REL_DIR!%%~nf_double.json"
    "%BINARY%" -i "%%f" -s %SOLVER% -p %PATCH_SIZE% --precision float --make-spd-from-pattern --out "%OUT_DIR%\!REL_DIR!%%~nf_float.json"
)

echo.
echo ============================================================
echo Done. Processed %COUNT% matrices
echo Results written to "%OUT_DIR%"
echo ============================================================

endlocal
