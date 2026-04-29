# IndexinGPU: Concurrent, Dynamic, Var-len Key/Value GPU Indexes

IndexinGPU is a series of GPU indexes that support variable-length key/values at runtime, fully GPU-managed dynamic insert/deletes, and concurrent mixes of different operation types.

It provides one ordered index (GPUMasstree) and three unordered indexes (GPUCuckooHT, GPUChainHT, GPUExtendHT).
Cuckoo and Chain are not resizable in runtime, but Extend is.

## Normal Build

IndexinGPU is tested on Ubuntu 24.04 and CUDA 13.2.

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

This builds unit tests `bin/unittest_*` and simple benchmarks `bin/*_bench`.

## Reproducing Experiment Results

Each evaluation takes ~1 day, so we recommend using `tmux new -s longjob`.

To evaluate GPU indexes (IndexinGPU + GPU baselines):

```shell
# Uses the first GPU (device_id=0)
# The GPU must have memory >= 80GiB
# Tested with docker image nvidia/cuda:13.2.1-devel-ubuntu24.04 and A100 GPU
apt update
apt install -y build-essential cmake git
git clone https://github.com/hyoungjook/IndexinGPU
cd IndexinGPU
bash evaluation/scripts/prepare_baselines.sh
bash evaluation/scripts/build_universal_bench_with_gpu.sh
python3 evaluation/scripts/measure_gpu.py --result-dir my-results
# This stores results in my-results/result_gpu.json
```

To evaluate CPU indexes (CPU baselines):

```shell
# Tested with docker image ubuntu:24.04
apt update
apt install -y build-essential cmake git autoconf libboost-all-dev libjemalloc-dev
git clone https://github.com/hyoungjook/IndexinGPU
cd IndexinGPU
bash evaluation/scripts/prepare_baselines.sh
bash evaluation/scripts/build_universal_bench_with_cpu_baseline.sh
python3 evaluation/scripts/measure_cpu.py --result-dir my-results
# This stores results in my-results/result_cpu.json
```

To draw plots:

```shell
# Move both result_gpu.json and result_cpu.json to my-results/
python3 evaluation/scripts/plot.py --result-dir my-results
```
