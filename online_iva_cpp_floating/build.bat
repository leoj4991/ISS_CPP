@echo off
REM Build separate_online.exe (fp32 build) with MinGW g++.
REM Everything needed is inside this folder: the FFTW single-precision header
REM and DLL live in third_party\, and the std::clamp / M_PI shim is
REM include\piva_online\compat.hpp (force-included below so it lands first).

setlocal
cd /d "%~dp0"

if "%CXX%"=="" set CXX=g++

%CXX% -std=c++17 -O3 -Wall -Wextra ^
  -Iinclude -Ithird_party ^
  -include include/piva_online/compat.hpp ^
  src\separate_online.cpp ^
  third_party\libfftw3f-3.dll ^
  -o separate_online.exe

if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)

REM the DLL has to sit next to the exe at run time
copy /y third_party\libfftw3f-3.dll libfftw3f-3.dll >nul

echo Built separate_online.exe
endlocal
