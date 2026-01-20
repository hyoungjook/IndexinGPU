/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *   Copyright 2025 Hyoungjoo Kim, Carnegie Mellon University
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
#pragma once

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <macros.hpp>

namespace GpuMasstree {
namespace kernels {
namespace cg = cooperative_groups;

template <typename masstree>
__global__ void initialize_kernel(masstree tree) {
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<masstree::cg_tile_size>(block);

  using allocator_type = typename masstree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  tree.allocate_root_node(tile, allocator);
}

template <typename device_func, typename masstree>
__global__ void batch_kernel(masstree tree,
                             const device_func func,
                             uint32_t num_requests) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<masstree::cg_tile_size>(block);

  using allocator_type = typename masstree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  uint32_t block_size = blockDim.x;
  uint32_t num_request_blocks = (num_requests + block_size - 1) / block_size;
  uint32_t num_worker_blocks = gridDim.x;
  for (uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
       thread_id < (num_request_blocks * block_size);
       thread_id += (num_worker_blocks * block_size)) {
    bool task_exists = (thread_id < num_requests);
    typename device_func::dev_regs regs;
    if (task_exists) { regs = func.load(thread_id, tile); }
    auto work_queue = tile.ballot(task_exists);
    
    while (work_queue) {
      int cur_rank = __ffs(work_queue) - 1;
      func.exec(tree, regs, tile, allocator, cur_rank);
      if (tile.thread_rank() == cur_rank) { task_exists = false; }
      work_queue = tile.ballot(task_exists);
    }

    if (thread_id < num_requests) { func.store(regs, thread_id); }
  }

}

template <typename key_slice_type, typename size_type, typename value_type>
struct insert_device_func {
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  const value_type* d_values;
  bool update_if_exists;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length,
      .value = d_values[thread_id]
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    tree.cooperative_insert(cur_key, cur_key_length, cur_value, tile, allocator, update_if_exists);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <typename key_slice_type, typename size_type, typename value_type>
struct find_device_func {
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tree.cooperative_find(cur_key, cur_key_length, tile, allocator, concurrent);
    if (tile.thread_rank() == cur_rank) {
      regs.value = cur_value;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_values[thread_id] = regs.value;
  }
};

template <bool do_merge, bool do_remove_empty_root, typename key_slice_type, typename size_type, typename value_type>
struct erase_device_func {
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length,
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    tree.cooperative_erase<do_merge, do_remove_empty_root>(cur_key, cur_key_length, tile, allocator, concurrent);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool use_upper_key, typename key_slice_type, typename size_type, typename value_type>
struct range_device_func {
  // kernel args
  const key_slice_type* d_lower_keys;
  const size_type* d_lower_key_lengths;
  size_type max_key_length;
  size_type max_count_per_query;
  const key_slice_type* d_upper_keys;
  const size_type* d_upper_key_lengths;
  size_type* d_counts;
  value_type* d_values;
  key_slice_type* d_out_keys;
  size_type* d_out_key_lengths;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* lower_key;
    size_type lower_key_length;
    const key_slice_type* upper_key;
    size_type upper_key_length;
    size_type count;
    value_type* value;
    key_slice_type* out_key;
    size_type* out_key_length;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .lower_key = &d_lower_keys[max_key_length * thread_id],
      .lower_key_length = d_lower_key_lengths ? d_lower_key_lengths[thread_id] : max_key_length,
      .upper_key = d_upper_keys ? &d_upper_keys[max_key_length * thread_id] : nullptr,
      .upper_key_length = d_upper_key_lengths ? d_upper_key_lengths[thread_id] : max_key_length,
      .value = d_values ? &d_values[max_count_per_query * thread_id] : nullptr,
      .out_key = d_out_keys ? &d_out_keys[max_count_per_query * max_key_length * thread_id] : nullptr,
      .out_key_length = d_out_key_lengths ? &d_out_key_lengths[max_count_per_query * thread_id] : nullptr
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, int cur_rank) const {
    auto cur_lower_key = tile.shfl(regs.lower_key, cur_rank);
    auto cur_lower_key_length = tile.shfl(regs.lower_key_length, cur_rank);
    auto cur_upper_key = tile.shfl(regs.upper_key, cur_rank);
    auto cur_upper_key_length = tile.shfl(regs.upper_key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    auto cur_out_key = tile.shfl(regs.out_key, cur_rank);
    auto cur_out_key_length = tile.shfl(regs.out_key_length, cur_rank);
    auto cur_count = tree.cooperative_range<use_upper_key>(
      cur_lower_key, cur_lower_key_length, tile, allocator, cur_upper_key, cur_upper_key_length,
      max_count_per_query, cur_value, cur_out_key, cur_out_key_length, max_key_length, concurrent);
    if (tile.thread_rank() == cur_rank) {
      regs.count = cur_count;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_counts[thread_id] = regs.count;
  }
};

template <bool do_merge, bool do_remove_empty_root, typename key_slice_type, typename value_type, typename size_type, typename masstree>
__global__ void test_insert_erase_kernel(const key_slice_type* insert_keys,
                                         const size_type* insert_key_lengths,
                                         const value_type* insert_values,
                                         const size_type insert_keys_count,
                                         const key_slice_type* erase_keys,
                                         const size_type* erase_key_lengths,
                                         const size_type erase_keys_count,
                                         const size_type max_key_length,
                                         masstree tree) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<masstree::cg_tile_size>(block);
  auto tile_id = thread_id / masstree::cg_tile_size;
  // if tile_id is even -> do insert. if tile_id is odd -> do erase.
  const bool is_insert = (tile_id % 2 == 0);
  // op_thread_id is id of threads within each op (insert/erase).
  auto op_thread_id = (tile_id / 2) * masstree::cg_tile_size + tile.thread_rank();
  if ((op_thread_id - tile.thread_rank()) >= (is_insert ? insert_keys_count : erase_keys_count)) { return; }
  const key_slice_type* key = nullptr;
  size_type key_length = 0;
  value_type value = masstree::invalid_value;
  bool to_do = false;
  if (is_insert) {
    if (op_thread_id < insert_keys_count) {
      key = &insert_keys[max_key_length * op_thread_id];
      key_length = insert_key_lengths ? insert_key_lengths[op_thread_id] : max_key_length;
      value = insert_values[op_thread_id];
      to_do = true;
    }
  }
  else {
    if (op_thread_id < erase_keys_count) {
      key = &erase_keys[max_key_length * op_thread_id];
      key_length = erase_key_lengths ? erase_key_lengths[op_thread_id] : max_key_length;
      to_do = true;
    }
  }
  using allocator_type = typename masstree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};
  auto work_queue = tile.ballot(to_do);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key = tile.shfl(key, cur_rank);
    auto cur_key_length = tile.shfl(key_length, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);
    if (is_insert) {
      tree.cooperative_insert(cur_key, cur_key_length, cur_value, tile, allocator);
    }
    else {
      tree.cooperative_erase<do_merge, do_remove_empty_root>(cur_key, cur_key_length, tile, allocator, true);
    }
    if (tile.thread_rank() == cur_rank) { to_do = false; }
    work_queue = tile.ballot(to_do);
  }
}

} // namespace kernels
} // namespace GpuMasstree
