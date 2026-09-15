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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <adapter_util.hpp>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <cooperative_groups.h>
#include <cuco/static_map.cuh>
#include <cuda/iterator>

template <typename key_type, typename value_type>
struct gpu_cuco_static_pair_generator {
  const key_type* keys;
  const value_type* values;

  __device__ cuco::pair<key_type, value_type> operator()(std::size_t i) const {
    return {keys[i], values[i]};
  }
};

template <typename ref_type, typename key_type, typename value_type>
__global__ void gpu_cuco_static_mixed_batch_kernel(
    const kernels::request_type* types,
    const key_type* keys,
    value_type* values,
    std::size_t num_keys,
    ref_type map_ref) {
  namespace cg = cooperative_groups;
  constexpr auto cg_size = ref_type::cg_size;
  auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
  auto request_idx =
    (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cg_size;
  if (request_idx >= num_keys) { return; }

  auto type = types[request_idx];
  auto key = keys[request_idx];
  if (type == kernels::request_type_insert) {
    map_ref.insert(tile, cuco::pair<key_type, value_type>{key, values[request_idx]});
  }
  else if (type == kernels::request_type_update) {
    map_ref.insert_or_assign(tile, cuco::pair<key_type, value_type>{key, values[request_idx]});
  }
  else if (type == kernels::request_type_erase) {
    map_ref.erase(tile, key);
  }
  else {
    auto found = map_ref.find(tile, key);
    if (tile.thread_rank() == 0) {
      values[request_idx] =
        found == map_ref.end() ? map_ref.empty_value_sentinel() : found->second;
    }
  }
}

struct gpu_cuco_static_adapter {
  static constexpr bool is_ordered = false;
  static constexpr bool support_mixed = true;
  static constexpr bool support_update = true;
  using key_slice_type = uint32_t;
  using value_slice_type = uint32_t;
  using size_type = uint32_t;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        using key_type = key_type_t<t.value>;
        using value_type = value_type_t<v.value>;
        auto capacity = static_cast<std::size_t>(
          std::ceil(configs_.num_keys / configs_.initial_array_fill_factor));
        index_ = new index_type<t.value, v.value>(
          capacity,
          cuco::empty_key<key_type>{std::numeric_limits<key_type>::max()},
          cuco::empty_value<value_type>{std::numeric_limits<value_type>::max()},
          cuco::erased_key<key_type>{std::numeric_limits<key_type>::max() - 1});
      });
    });
  }
  void destroy() {
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        delete reinterpret_cast<index_type<t.value, v.value>*>(index_);
      });
    });
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_slice_type* values,
              uint32_t valuelen_max,
              const size_type* value_lengths,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)value_lengths;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        using key_type = key_type_t<t.value>;
        using value_type = value_type_t<v.value>;
        auto typed_keys = reinterpret_cast<const key_type*>(keys);
        auto typed_values = reinterpret_cast<const value_type*>(values);
        auto pairs = cuda::make_transform_iterator(
          cuda::counting_iterator<std::size_t>{0},
          gpu_cuco_static_pair_generator<key_type, value_type>{typed_keys, typed_values});
        get_index<t.value, v.value>()->insert_async(pairs, pairs + num_keys);
      });
    });
  }
  void update(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_slice_type* values,
              uint32_t valuelen_max,
              const size_type* value_lengths,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)value_lengths;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        using key_type = key_type_t<t.value>;
        using value_type = value_type_t<v.value>;
        auto typed_keys = reinterpret_cast<const key_type*>(keys);
        auto typed_values = reinterpret_cast<const value_type*>(values);
        auto pairs = cuda::make_transform_iterator(
          cuda::counting_iterator<std::size_t>{0},
          gpu_cuco_static_pair_generator<key_type, value_type>{typed_keys, typed_values});
        get_index<t.value, v.value>()->insert_or_assign_async(pairs, pairs + num_keys);
      });
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        auto typed_keys = reinterpret_cast<const key_type_t<t.value>*>(keys);
        get_index<t.value, v.value>()->erase_async(typed_keys, typed_keys + num_keys);
      });
    });
  }
  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_slice_type* results,
            uint32_t valuelen_max,
            size_type* result_lengths,
            std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)result_lengths;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        auto typed_keys = reinterpret_cast<const key_type_t<t.value>*>(keys);
        auto typed_results = reinterpret_cast<value_type_t<v.value>*>(results);
        get_index<t.value, v.value>()->find_async(
          typed_keys, typed_keys + num_keys, typed_results);
      });
    });
  }
  void mixed_batch(const kernels::request_type* types,
                   const key_slice_type* keys,
                   uint32_t keylen_max,
                   const size_type* key_lengths,
                   value_slice_type* values,
                   uint32_t valuelen_max,
                   size_type* value_lengths,
                   std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    (void)valuelen_max;
    (void)value_lengths;
    if (num_keys == 0) { return; }
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        using key_type = key_type_t<t.value>;
        using value_type = value_type_t<v.value>;
        constexpr uint32_t block_size = 128;
        constexpr uint32_t cg_size = index_type<t.value, v.value>::cg_size;
        auto num_blocks = static_cast<uint32_t>(
          (num_keys * cg_size + block_size - 1) / block_size);
        auto typed_keys = reinterpret_cast<const key_type*>(keys);
        auto typed_values = reinterpret_cast<value_type*>(values);
        auto map_ref = get_index<t.value, v.value>()->ref(
          cuco::insert, cuco::insert_or_assign, cuco::erase, cuco::find);
        gpu_cuco_static_mixed_batch_kernel<<<num_blocks, block_size>>>(
          types, typed_keys, typed_values, num_keys, map_ref);
      });
    });
  }
  void print_stats() {}
  void ht_print_load_factor(std::size_t max_keys, uint32_t key_length, uint32_t value_length) {
    (void)key_length;
    (void)value_length;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      adapter_util::dispatch_uint32<1, 2>(configs_.valuelen_max, [&](auto v) {
        std::cout << "LoadFactor: "
                  << static_cast<double>(max_keys) /
                       get_index<t.value, v.value>()->capacity()
                  << std::endl;
      });
    });
  }

 private:
  #define FORALL_ARGUMENTS_GPU_CUCO_STATIC(x) \
    x(initial_array_fill_factor, float, 0.8f)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_GPU_CUCO_STATIC(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    std::size_t num_keys;
    uint32_t keylen_max;
    uint32_t valuelen_max;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_GPU_CUCO_STATIC(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      #define PARSE_DEFAULT_ARGUMENTS(arg, type, default_value) \
      [[maybe_unused]] auto tmp_##arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS(PARSE_DEFAULT_ARGUMENTS)
      #undef PARSE_DEFAULT_ARGUMENTS
      num_keys = tmp_max_keys;
      keylen_max = tmp_keylen_max;
      valuelen_max = tmp_valuelen_max;
      check_argument(tmp_keylen_min == tmp_keylen_max);
      check_argument(tmp_valuelen_min == tmp_valuelen_max);
      check_argument((tmp_keylen_max == 1 &&
                      (tmp_valuelen_max == 1 || tmp_valuelen_max == 2)) ||
                     (tmp_keylen_max == 2 && tmp_valuelen_max == 1));
      check_argument(0 < initial_array_fill_factor && initial_array_fill_factor < 1.0f);
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_GPU_CUCO_STATIC(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_GPU_CUCO_STATIC

  template <uint32_t key_length>
  using key_type_t = std::conditional_t<key_length == 1, uint32_t, uint64_t>;
  template <uint32_t value_length>
  using value_type_t = std::conditional_t<value_length == 1, uint32_t, uint64_t>;
  template <uint32_t key_length, uint32_t value_length>
  using index_type = cuco::static_map<key_type_t<key_length>, value_type_t<value_length>>;

  template <uint32_t key_length, uint32_t value_length>
  index_type<key_length, value_length>* get_index() {
    return reinterpret_cast<index_type<key_length, value_length>*>(index_);
  }

  configs configs_;
  void* index_;
};
