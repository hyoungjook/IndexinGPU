# IndexinGPU

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
# EXPERIMENTS TUNED FOR GPU WITH MEMORY >= 80GiB
# Uses the first GPU (device_id=0)
# Assume Ubuntu 24.04 and CUDA 13.2
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
# Assume Ubuntu 24.04, no CUDA required
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

## Verifying Cache Line Atomicity

IndexinGPU relies on Nvidia GPU's cache line atomicity behavior.
Use this [cuda_cacheline_atomicity_tester](https://github.com/hyoungjook/cuda_cacheline_atomicity_tester) to test your GPU shows cache line atomicity.
Verified on A100, H100, H200, and B200 GPUs.
