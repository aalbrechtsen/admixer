#!/usr/bin/env bash
# Build a portable admixer binary (Linux x86_64, glibc >= 2.28) inside the manylinux_2_28 container (GCC 14).
# OpenBLAS (DYNAMIC_ARCH: picks kernels for the CPU at runtime) and zlib are built from source and linked
# statically, as are libstdc++/libgcc/libgomp. Output: dist/admixer-linux-x86_64.tar.gz
# Usage: ./build-static.sh   (needs podman or docker; the dependency builds are cached in .build/)
set -euo pipefail
cd "$(dirname "$0")"
OPENBLAS_VER=${OPENBLAS_VER:-0.3.28}
ZLIB_VER=${ZLIB_VER:-1.3.1}
REL_ARCH=${REL_ARCH:-"-march=x86-64-v2 -mtune=generic"}
IMAGE=quay.io/pypa/manylinux_2_28_x86_64
ENGINE=$(command -v podman || command -v docker)
mkdir -p .build dist

"$ENGINE" run --rm -v "$PWD":/src:Z -w /src -e OPENBLAS_VER="$OPENBLAS_VER" -e ZLIB_VER="$ZLIB_VER" \
  -e REL_ARCH="$REL_ARCH" "$IMAGE" bash -euc '
  D=/src/.build/deps
  if [ ! -f $D/lib/libopenblas.a ]; then
    mkdir -p /src/.build/src && cd /src/.build/src
    curl -sSL https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OPENBLAS_VER/OpenBLAS-$OPENBLAS_VER.tar.gz | tar xz --no-same-owner
    cd OpenBLAS-$OPENBLAS_VER
    make -j"$(nproc)" DYNAMIC_ARCH=1 TARGET=NEHALEM NO_LAPACK=1 NO_FORTRAN=1 NO_SHARED=1 \
         USE_OPENMP=1 NUM_THREADS=256 libs >/dev/null 2>&1
    make PREFIX=$D NO_LAPACK=1 NO_FORTRAN=1 NO_SHARED=1 install >/dev/null
  fi
  if [ ! -f $D/lib/libz.a ]; then
    mkdir -p /src/.build/src && cd /src/.build/src
    curl -sSL https://github.com/madler/zlib/releases/download/v$ZLIB_VER/zlib-$ZLIB_VER.tar.gz | tar xz --no-same-owner
    cd zlib-$ZLIB_VER
    CFLAGS="-O3 -fPIC" ./configure --static --prefix=$D >/dev/null
    make -j"$(nproc)" install >/dev/null
  fi
  cd /src
  make release CXX=g++ REL_ARCH="$REL_ARCH" OPENBLAS_A=$D/lib/libopenblas.a ZLIB_A=$D/lib/libz.a DEP_INC=-I$D/include
  strip admixer
'

echo "Runtime dependencies:"
ldd admixer
if ldd admixer | grep -vE "linux-vdso|ld-linux|lib(c|m|pthread|dl|rt)\.so" | grep -q "=>"; then
  echo "ERROR: unexpected shared library dependency" >&2; exit 1
fi
tar czf dist/admixer-linux-x86_64.tar.gz admixer README.md LICENSE
echo "Wrote dist/admixer-linux-x86_64.tar.gz"
