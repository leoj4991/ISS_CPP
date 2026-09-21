#!/bin/sh
# Build separate_online. Needs a C++17 compiler and FFTW3.
set -e
cd "$(dirname "$0")"

CXX=${CXX:-c++}
FLAGS=""

# Find FFTW: Homebrew (macOS), then pkg-config, else the default system paths (Linux)
if command -v brew >/dev/null 2>&1 && [ -f "$(brew --prefix)/include/fftw3.h" ]; then
  P=$(brew --prefix)
  FLAGS="-I$P/include -L$P/lib -Wl,-rpath,$P/lib"
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists fftw3; then
  FLAGS=$(pkg-config --cflags --libs-only-L fftw3)
fi

$CXX -std=c++17 -O3 -Wall -Wextra -Iinclude src/separate_online.cpp $FLAGS -lfftw3 -o separate_online
echo "Built ./separate_online"
