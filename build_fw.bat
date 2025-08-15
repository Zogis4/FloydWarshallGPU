@echo off
setlocal enabledelayedexpansion

rem ==========================================
rem  Build Floyd–Warshall (Windows, VS + CUDA)
rem  Use no "x64 Native Tools Command Prompt for VS 2022"
rem ==========================================

set SM=89
set OMP=/openmp:llvm
set OPT=/O2 /std:c11

rem fontes
set SRC_CPU=fw_impl_cpu.c
set SRC_CUDA=fw_impl_cuda.cu
set SRC_MAIN_BENCH=fw_bench_main.cu

rem saida
set EXE_BENCH=fwbench.exe

if /I "%~1"=="clean" goto :CLEAN

echo === COMPILA CPU (MSVC) ===
cl %OPT% %OMP% /c "%SRC_CPU%"
if errorlevel 1 goto :ERR

echo === COMPILA E LINKA CUDA (NVCC) ===
where nvcc >nul 2>&1
if errorlevel 1 (
  echo [ERRO] nvcc nao encontrado. Abra o "x64 Native Tools..." do VS ou instale o CUDA Toolkit.
  goto :ERR
)

nvcc -O3 -std=c++14 ^
  -gencode arch=compute_%SM%,code=sm_%SM% ^
  -gencode arch=compute_%SM%,code=compute_%SM% ^
  "%SRC_CUDA%" "%SRC_MAIN_BENCH%" fw_impl_cpu.obj ^
  -Xcompiler "%OMP%" ^
  -o "%EXE_BENCH%"
if errorlevel 1 goto :ERR

echo [OK] Gerado: %EXE_BENCH%
echo.
echo Exemplo de execucao:
echo   %EXE_BENCH% --ns 256,512,768,1024 --dens 25,50,75 --reps 5 --omp-threads 16
goto :END

:CLEAN
del /q *.obj 2>nul
del /q "%EXE_BENCH%" 2>nul
echo [OK] Limpo.
goto :END

:ERR
echo.
echo [FALHA] Veja as mensagens acima.
exit /b 1

:END
endlocal
