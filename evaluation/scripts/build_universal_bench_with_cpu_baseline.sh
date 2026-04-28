#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX_BIN="${CXX:-c++}"
CC_BIN="${CC:-cc}"
OUTPUT_PATH="${OUTPUT:-${ROOT_DIR}/build/bin/universal_bench_with_cpu_baseline}"
MASSTREE_DIR="${ROOT_DIR}/baselines/masstree-beta"
ARTSYNC_DIR="${ROOT_DIR}/baselines/ARTSynchronized"
FOLLY_DIR="${ROOT_DIR}/baselines/folly"

if [[ -n "${TBB_LDFLAGS:-}" ]]; then
  read -r -a TBB_LDFLAGS_ARR <<< "${TBB_LDFLAGS}"
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists tbb; then
  read -r -a TBB_LDFLAGS_ARR <<< "$(pkg-config --libs tbb)"
else
  TBB_LDFLAGS_ARR=(-ltbb)
fi

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

ROWEX_SOURCES=(
  "${ARTSYNC_DIR}/ROWEX/Tree.cpp"
)

FOLLY_SOURCES=(
  "${ROOT_DIR}/evaluation/cpu_folly_support.cpp"
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
  -std=c++20 \
  -O3 \
  -pthread \
  -include "${MASSTREE_DIR}/config.h" \
  -include cstring \
  -I"${ROOT_DIR}/evaluation" \
  -I"${ROOT_DIR}/evaluation/folly_compat" \
  -I"${ROOT_DIR}/include" \
  -I"${ROOT_DIR}/baselines/libcuckoo" \
  -I"${ROOT_DIR}/baselines/oneTBB/include" \
  -I"${FOLLY_DIR}" \
  -I"${MASSTREE_DIR}" \
  -I"${ARTSYNC_DIR}" \
  -DUNIVERSAL_BENCH_WITH_CPU_BASELINE \
  -DNOGPU \
  -DFOLLY_NO_CONFIG \
  -DFOLLY_MOBILE=1 \
  -x c++ \
  "${ROOT_DIR}/evaluation/universal_bench.cu" \
  "${MASSTREE_SOURCES[@]}" \
  "${ROWEX_SOURCES[@]}" \
  "${FOLLY_SOURCES[@]}" \
  -o "${OUTPUT_PATH}" \
  -lm \
  "${TBB_LDFLAGS_ARR[@]}" \
  "$@"

echo "Built ${OUTPUT_PATH}"
