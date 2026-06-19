# IndexinGPU

IndexinGPU is a header-only CUDA/C++ library of GPU-resident indexes for workloads that need more than fixed-width key/value lookups. It is designed for:

- variable-length keys and values
- fully GPU-managed inserts and deletes
- concurrent mixed lookup/insert/delete workloads
- both ordered and unordered indexing on the GPU

The library currently provides:

- `GpuMasstree::gpu_masstree`: ordered index with point lookup, insert, erase, scan, and mixed batches
- `GpuHashtable::gpu_cuckoohashtable`: fixed-capacity unordered index
- `GpuHashtable::gpu_chainhashtable`: fixed-capacity unordered index
- `GpuExtendHashtable::gpu_extendhashtable`: resizable unordered index

For most users, the main entry points are the headers in [`include/`](/Users/hyoungjoo/workspace/IndexinGPU/include).

Source code started from [MVGpuBTree](https://github.com/owensgroup/MVGpuBTree).

## Build

Tested on Ubuntu 24.04 and CUDA 13.2.

Need cmake >= 3.24.

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

This builds the unit tests in `bin/unittest_*` and the simple benchmarks in `bin/*_bench`.

If you use this repo as a CMake subproject, link against `indexingpu`.

## CPU-side API

The CPU-side API is the host control plane: construct an index on the host, then launch batched operations over device buffers.

All four indexes use the same basic layout:

- keys are flat arrays of `uint32_t` slices with shape `[num_requests * max_key_length]`
- values are flat arrays of `uint32_t` slices with shape `[num_requests * max_value_length]`
- `key_lengths` is optional; passing `nullptr` means "treat every key as fixed-length using `max_key_length`"
- insert-side `value_lengths` is also optional; passing `nullptr` means "treat every value as fixed-length using `max_value_length`"
- lookup and scan outputs write their result lengths to `value_lengths`

For lookups, `value_lengths[i] == 0` means the key was not found.

```cpp
#include <gpu_extendhashtable.hpp>
#include <simple_slab_linear_alloc.hpp>
#include <simple_debra_reclaim.hpp>

// use simple_slab_allocator<128> for others
using allocator_type = simple_slab_linear_allocator<128>;
using reclaimer_type = simple_debra_reclaimer<>;
using index_type =
    GpuExtendHashtable::gpu_extendhashtable<allocator_type, reclaimer_type, 16>;

allocator_type allocator(0.9f);
reclaimer_type reclaimer;
// User can instantiate multiple indexes with common allocator/reclaimer
// except only one gpu_extendhashtable is allowed.
index_type index(allocator, reclaimer, 1024, 2.0f, 2.5f);

index.insert(d_keys, max_key_length, d_key_lengths,
             d_values, max_value_length, d_value_lengths,
             num_keys);

index.find(d_keys, max_key_length, d_key_lengths,
           d_out_values, max_value_length, d_out_value_lengths,
           num_keys);

index.erase(d_keys, max_key_length, d_key_lengths, num_keys);
```

Common host methods:

- `find(..., cudaStream_t stream = 0)`
- `insert(..., cudaStream_t stream = 0, bool update_if_exists = false)`
- `erase(..., cudaStream_t stream = 0)`
- `mixed_batch(..., bool* results, ..., cudaStream_t stream = 0, bool insert_update_if_exists = false)`
- `scan(...)` on `gpu_masstree` only

Constructor differences:

- `gpu_masstree(allocator, reclaimer)`
- `gpu_chainhashtable(allocator, reclaimer, expected_elements, fill_factor)` or explicit bucket count
- `gpu_cuckoohashtable(allocator, reclaimer, expected_elements, fill_factor)` or explicit buckets-per-hash-function count
- `gpu_extendhashtable(allocator, reclaimer, initial_directory_size, resize_policy, load_factor_threshold)`

For mixed batches, use `kernels::request_type_insert`, `kernels::request_type_find`, and `kernels::request_type_erase`. Lookup requests write back `values` and `value_lengths`; insert and erase requests report success through `results` when provided.

`gpu_masstree` also supports range scans from the host:

```cpp
tree.scan(lower_keys, lower_key_lengths,
          max_key_length, max_count_per_query, num_queries,
          upper_keys, upper_key_lengths,
          counts, out_values, max_value_length, out_value_lengths,
          out_keys, out_key_lengths);
```

For end-to-end host-side examples, see [`test/`](/Users/hyoungjoo/workspace/IndexinGPU/test) and [`benchmarks/`](/Users/hyoungjoo/workspace/IndexinGPU/benchmarks).

## GPU-side API

Each index also exposes cooperative device-side methods so you can call the index directly inside your own CUDA kernels. These are lower-level building blocks than the host batch API.

Read-only lookups follow this pattern:

```cpp
template <class Index>
__global__ void probe_kernel(Index index,
                             const uint32_t* keys,
                             const uint32_t* key_lengths,
                             uint32_t* values,
                             uint32_t* value_lengths,
                             uint32_t max_key_length,
                             uint32_t max_value_length,
                             uint32_t num_keys) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<Index::tile_size_>(block);

  uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / Index::tile_size_;
  if (i >= num_keys) return;

  typename Index::device_allocator_context_type allocator{index.allocator_, tile};
  const uint32_t* key = &keys[i * max_key_length];
  uint32_t* value = &values[i * max_value_length];
  uint32_t key_length = key_lengths ? key_lengths[i] : max_key_length;

  auto value_length =
      index.template cooperative_find<false>(key, key_length, value,
                                             max_value_length, tile, allocator);

  if (tile.thread_rank() == 0) {
    value_lengths[i] = value_length;
  }
}
```

The main device-side entry points are:

- `cooperative_find(...)`
- `cooperative_insert(...)`
- `cooperative_erase(...)`
- `cooperative_scan(...)` on `gpu_masstree`

Important details:

- device-side calls are cooperative and expect a `cooperative_groups` tile of `16` or `32` threads, matching the index template parameter
- read-only calls need a `device_allocator_context_type`
- inserts and erases also need a `device_reclaimer_context_type`

If you just want batched operations on arrays of keys, prefer the CPU-side API above. If you want to fuse index operations into a larger CUDA kernel, use the cooperative device API.

## Hardware Note

IndexinGPU relies on NVIDIA GPU cache-line atomicity behavior. It has been verified on A100, H100, H200, and B200 GPUs.

To check this behavior on your hardware, see [cuda_cacheline_atomicity_tester](https://github.com/hyoungjook/cuda_cacheline_atomicity_tester).

## Reproducing Experiment Results

Each evaluation takes 1-2 days, so we recommend using `tmux new -s longjob`.

Before evaluation, download the [Memetracker dataset](https://snap.stanford.edu/memetracker/data.html), extract P entries (URLs), filter out URLs longer than 128B, and store it into an ascii txt file with the string URLs separated by line breaks.
The result file should have 91114167 entries.

If you want to evaluate only with synthetic workloads, add `--skip-meme` command line option to all `python3` commands below.

To evaluate GPU indexes (IndexinGPU + GPU baselines):

```shell
# EXPERIMENTS TUNED FOR GPU WITH MEMORY >= 80GiB
# Uses the first GPU (device_id=0)
# Assume Ubuntu 24.04 and CUDA 13.2
apt update
apt install -y build-essential cmake git
git clone https://github.com/hyoungjook/IndexinGPU
cd IndexinGPU
# copy memetracker dataset
mkdir dataset
cp path/to/memetracker/dataset dataset/meme.txt
# evaluate
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
# copy memetracker dataset
mkdir dataset
cp path/to/memetracker/dataset dataset/meme.txt
# evaluate
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
