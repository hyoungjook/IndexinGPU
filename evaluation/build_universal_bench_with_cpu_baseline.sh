#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX_BIN="${CXX:-c++}"
OUTPUT_PATH="${OUTPUT:-${ROOT_DIR}/bin/universal_bench_with_cpu_baseline}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

"${CXX_BIN}" \
  -std=c++17 \
  -O3 \
  -pthread \
  -I"${ROOT_DIR}/evaluation" \
  -I"${ROOT_DIR}/include" \
  -I"${ROOT_DIR}/baselines/libcuckoo" \
  "${ROOT_DIR}/evaluation/universal_bench_cpu.cpp" \
  -o "${OUTPUT_PATH}" \
  "$@"

echo "Built ${OUTPUT_PATH}"
