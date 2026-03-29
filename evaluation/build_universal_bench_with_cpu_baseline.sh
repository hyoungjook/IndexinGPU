#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX_BIN="${CXX:-c++}"
CC_BIN="${CC:-cc}"
OUTPUT_PATH="${OUTPUT:-${ROOT_DIR}/bin/universal_bench_with_cpu_baseline}"
MASSTREE_DIR="${ROOT_DIR}/baselines/masstree-beta"

MASSTREE_SOURCES=(
  "${ROOT_DIR}/evaluation/cpu_masstree_support.cpp"
  "${MASSTREE_DIR}/compiler.cc"
  "${MASSTREE_DIR}/misc.cc"
  "${MASSTREE_DIR}/json.cc"
  "${MASSTREE_DIR}/string.cc"
  "${MASSTREE_DIR}/straccum.cc"
  "${MASSTREE_DIR}/str.cc"
  "${MASSTREE_DIR}/msgpack.cc"
  "${MASSTREE_DIR}/clp.c"
  "${MASSTREE_DIR}/kvrandom.cc"
  "${MASSTREE_DIR}/memdebug.cc"
  "${MASSTREE_DIR}/kvthread.cc"
)

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if [[ ! -f "${MASSTREE_DIR}/config.h" ]]; then
  if [[ ! -x "${MASSTREE_DIR}/configure" ]]; then
    (
      cd "${MASSTREE_DIR}"
      ./bootstrap.sh
    )
  fi
  (
    cd "${MASSTREE_DIR}"
    ./configure CXX="${CXX_BIN}" CC="${CC_BIN}"
  )
fi

"${CXX_BIN}" \
  -std=c++17 \
  -O3 \
  -pthread \
  -include "${MASSTREE_DIR}/config.h" \
  -I"${ROOT_DIR}/evaluation" \
  -I"${ROOT_DIR}/include" \
  -I"${ROOT_DIR}/baselines/libcuckoo" \
  -I"${MASSTREE_DIR}" \
  "${ROOT_DIR}/evaluation/universal_bench_cpu.cpp" \
  "${MASSTREE_SOURCES[@]}" \
  -o "${OUTPUT_PATH}" \
  -lm \
  "$@"

echo "Built ${OUTPUT_PATH}"
