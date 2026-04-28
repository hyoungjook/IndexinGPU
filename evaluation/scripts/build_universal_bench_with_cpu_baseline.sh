#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX_BIN="${CXX:-c++}"
CC_BIN="${CC:-cc}"
OUTPUT_PATH="${OUTPUT:-${ROOT_DIR}/build/bin/universal_bench_with_cpu_baseline}"
MASSTREE_DIR="${ROOT_DIR}/baselines/masstree-beta"
ARTSYNC_DIR="${ROOT_DIR}/baselines/ARTSynchronized"
ONETBB_DIR="${ROOT_DIR}/baselines/oneTBB"
ONETBB_BUILD_DIR="${ROOT_DIR}/build/onetbb"
ONETBB_INSTALL_DIR="${ONETBB_BUILD_DIR}/install"
ONETBB_MANUAL_BUILD_DIR="${ONETBB_BUILD_DIR}/manual"
ONETBB_MANUAL_LIB_DIR="${ONETBB_MANUAL_BUILD_DIR}/lib"
ONETBB_MANUAL_OBJ_DIR="${ONETBB_MANUAL_BUILD_DIR}/obj"
ONETBB_INCLUDE_DIR="${ONETBB_INSTALL_DIR}/include"
ONETBB_CMAKE_BIN="${CMAKE:-cmake}"
ONETBB_LIB_DIRS=()

ensure_onetbb() {
  if ! command -v "${ONETBB_CMAKE_BIN}" >/dev/null 2>&1; then
    echo "ERROR: cmake not found."
    exit 0
  fi

  "${ONETBB_CMAKE_BIN}" \
    -S "${ONETBB_DIR}" \
    -B "${ONETBB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ONETBB_INSTALL_DIR}" \
    -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
    -DCMAKE_C_COMPILER="${CC_BIN}" \
    -DTBB_TEST=OFF \
    -DTBB_EXAMPLES=OFF \
    -DTBB_STRICT=OFF \
    -DTBBMALLOC_PROXY_BUILD=OFF

  local build_parallelism
  build_parallelism="${BUILD_PARALLELISM:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}"
  "${ONETBB_CMAKE_BIN}" --build "${ONETBB_BUILD_DIR}" --config Release --target install --parallel "${build_parallelism}"
  ONETBB_INCLUDE_DIR="${ONETBB_INSTALL_DIR}/include"
  ONETBB_LIB_DIRS=("${ONETBB_INSTALL_DIR}/lib" "${ONETBB_INSTALL_DIR}/lib64")
}

find_onetbb_libs() {
  local lib_dir
  if [[ "${#ONETBB_LIB_DIRS[@]}" -eq 0 ]]; then
    ONETBB_LIB_DIRS=("${ONETBB_MANUAL_LIB_DIR}" "${ONETBB_INSTALL_DIR}/lib" "${ONETBB_INSTALL_DIR}/lib64")
  fi
  for lib_dir in "${ONETBB_LIB_DIRS[@]}"; do
    if [[ -e "${lib_dir}/libtbb.so" || -e "${lib_dir}/libtbb.dylib" ]]; then
      ONETBB_LIB_DIR="${lib_dir}"
      break
    fi
  done

  if [[ -z "${ONETBB_LIB_DIR:-}" ]]; then
    echo "Unable to find vendored oneTBB libraries under ${ONETBB_INSTALL_DIR}." >&2
    return 1
  fi

  if [[ -e "${ONETBB_LIB_DIR}/libtbb.so" ]]; then
    ONETBB_TBB_LIB="${ONETBB_LIB_DIR}/libtbb.so"
    ONETBB_MALLOC_LIB="${ONETBB_LIB_DIR}/libtbbmalloc.so"
  else
    ONETBB_TBB_LIB="${ONETBB_LIB_DIR}/libtbb.dylib"
    ONETBB_MALLOC_LIB="${ONETBB_LIB_DIR}/libtbbmalloc.dylib"
  fi
}

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

mkdir -p "$(dirname "${OUTPUT_PATH}")"
ensure_onetbb
find_onetbb_libs

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
  -DNDEBUG \
  -pthread \
  -include "${MASSTREE_DIR}/config.h" \
  -include cstring \
  -I"${ROOT_DIR}/evaluation" \
  -I"${ROOT_DIR}/include" \
  -I"${ROOT_DIR}/baselines/libcuckoo" \
  -I"${ONETBB_INCLUDE_DIR}" \
  -I"${MASSTREE_DIR}" \
  -I"${ARTSYNC_DIR}" \
  -DUNIVERSAL_BENCH_WITH_CPU_BASELINE \
  -DNOGPU \
  -x c++ \
  "${ROOT_DIR}/evaluation/universal_bench.cu" \
  "${MASSTREE_SOURCES[@]}" \
  "${ROWEX_SOURCES[@]}" \
  -o "${OUTPUT_PATH}" \
  -lm \
  -Wl,-rpath,"${ONETBB_LIB_DIR}" \
  -Wl,-rpath-link,"${ONETBB_LIB_DIR}" \
  "${ONETBB_TBB_LIB}" \
  -Wl,--no-as-needed \
  "${ONETBB_MALLOC_LIB}" \
  -Wl,--as-needed \
  "$@"

echo "Built ${OUTPUT_PATH}"
