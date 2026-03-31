#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target universal_bench --config Release
cmake --build "${BUILD_DIR}" --target universal_bench_with_gpu_baseline --config Release
