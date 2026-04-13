#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DYCUCKOO_DIR="${ROOT_DIR}/baselines/DyCuckoo"
DYCUCKOO_PATCH="${ROOT_DIR}/baselines/DyCuckoo.patch"
MASSTREE_DIR="${ROOT_DIR}/baselines/masstree-beta"
MASSTREE_PATCH="${ROOT_DIR}/baselines/masstree-beta.patch"
BTREE_DIR="${ROOT_DIR}/baselines/MVGpuBtree"
BTREE_PATCH="${ROOT_DIR}/baselines/MVGpuBtree.patch"

pushd $(pwd) > /dev/null
cd ${ROOT_DIR} && git submodule update --init
cd ${DYCUCKOO_DIR} && git apply ${DYCUCKOO_PATCH}
cd ${MASSTREE_DIR} && git apply ${MASSTREE_PATCH}
cd ${BTREE_DIR} && git apply ${BTREE_PATCH}
popd > /dev/null
