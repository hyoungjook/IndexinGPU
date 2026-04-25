/*
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <device_context.hpp>
#include <macros.hpp>
#include <pair_type.hpp>

namespace GpuBTree {
template <typename Key, typename Value, int b = 16>
struct node_type {
  using T = pair_type<Key, Value>;
  T node[b];
};
}  // namespace GpuBTree
#include <gpu_blink_tree.hpp>

struct gpu_blink_tree_adapter {
  static constexpr bool is_ordered = true;
  static constexpr bool support_mixed = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr uint32_t tile_size = 16;
  using node_type = GpuBTree::node_type<key_slice_type, value_type, tile_size>;
  static constexpr std::size_t gpu_memory_capacity = 80ULL * 1024 * 1024 * 1024;
  static constexpr std::size_t allocator_pool_bytes = static_cast<std::size_t>(0.9 * gpu_memory_capacity);
  static constexpr std::size_t allocator_max_count = allocator_pool_bytes / sizeof(node_type);
  using allocator_type = device_bump_allocator<node_type, allocator_max_count>;
  using index_type = GpuBTree::gpu_blink_tree<key_slice_type, value_type, tile_size, allocator_type>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    index_ = new index_type();
  }
  void destroy() {
    delete index_;
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              uint32_t valuelen_max,
              const size_type* value_lengths,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)value_lengths;
    index_->insert(keys, values, num_keys);
  }

  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    index_->erase(keys, num_keys, 0, configs_.erase_concurrent);
  }

  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            uint32_t valuelen_max,
            size_type* result_lengths,
            std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)result_lengths;
    index_->find(keys, results, num_keys, 0, configs_.lookup_concurrent);
  }

  void scan(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            uint32_t count,
            value_type* results,
            uint32_t valuelen_max,
            size_type* result_lengths,
            std::size_t num_keys,
            const key_slice_type* upper_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)result_lengths;
    index_->range_query(keys, upper_keys,
      reinterpret_cast<pair_type<key_slice_type, value_type>*>(results), nullptr,
      count, num_keys, 0, configs_.lookup_concurrent);
  }
  void mixed_batch(const kernels::request_type* types,
                   const key_slice_type* keys,
                   uint32_t keylen_max,
                   const size_type* key_lengths,
                   value_type* values,
                   uint32_t valuelen_max,
                   size_type* value_lengths,
                   std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)value_lengths;
    index_->mixed_batch(types, keys, values, num_keys);
  }
  void print_stats() {}

 private:
  #define FORALL_ARGUMENTS_GPU_BLINKTREE(x) \
    x(lookup_concurrent, bool, false) \
    x(erase_concurrent, bool, false)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_GPU_BLINKTREE(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_GPU_BLINKTREE(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      #define PARSE_DEFAULT_ARGUMENTS(arg, type, default_value) \
      [[maybe_unused]] auto arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS(PARSE_DEFAULT_ARGUMENTS)
      #undef PARSE_DEFAULT_ARGUMENTS
      check_argument(keylen_max == 1);
      check_argument(valuelen_max == 1);
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_GPU_BLINKTREE(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_GPU_BLINKTREE

  configs configs_;
  index_type* index_;
};
