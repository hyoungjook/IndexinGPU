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

struct gpu_cuco_static_adapter {
  static constexpr bool is_ordered = false;
  static constexpr bool support_mixed = false;
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
      using key_type = key_type_t<t.value>;
      auto capacity = static_cast<std::size_t>(
        std::ceil(configs_.num_keys / configs_.initial_array_fill_factor));
      index_ = new index_type<t.value>(
        capacity,
        cuco::empty_key<key_type>{std::numeric_limits<key_type>::max()},
        cuco::empty_value<value_slice_type>{std::numeric_limits<value_slice_type>::max()},
        cuco::erased_key<key_type>{std::numeric_limits<key_type>::max() - 1});
    });
  }
  void destroy() {
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      delete reinterpret_cast<index_type<t.value>*>(index_);
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
      using key_type = key_type_t<t.value>;
      auto typed_keys = reinterpret_cast<const key_type*>(keys);
      auto pairs = cuda::make_transform_iterator(
        cuda::counting_iterator<std::size_t>{0},
        gpu_cuco_static_pair_generator<key_type, value_slice_type>{typed_keys, values});
      get_index<t.value>()->insert_async(pairs, pairs + num_keys);
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
      using key_type = key_type_t<t.value>;
      auto typed_keys = reinterpret_cast<const key_type*>(keys);
      auto pairs = cuda::make_transform_iterator(
        cuda::counting_iterator<std::size_t>{0},
        gpu_cuco_static_pair_generator<key_type, value_slice_type>{typed_keys, values});
      get_index<t.value>()->insert_or_assign_async(pairs, pairs + num_keys);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      auto typed_keys = reinterpret_cast<const key_type_t<t.value>*>(keys);
      get_index<t.value>()->erase_async(typed_keys, typed_keys + num_keys);
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
      auto typed_keys = reinterpret_cast<const key_type_t<t.value>*>(keys);
      get_index<t.value>()->find_async(typed_keys, typed_keys + num_keys, results);
    });
  }
  void print_stats() {}
  void ht_print_load_factor(std::size_t max_keys, uint32_t key_length, uint32_t value_length) {
    (void)key_length;
    (void)value_length;
    adapter_util::dispatch_uint32<1, 2>(configs_.keylen_max, [&](auto t) {
      std::cout << "LoadFactor: "
                << static_cast<double>(max_keys) / get_index<t.value>()->capacity()
                << std::endl;
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
      check_argument(tmp_keylen_min == tmp_keylen_max);
      check_argument(tmp_keylen_max == 1 || tmp_keylen_max == 2);
      check_argument(tmp_valuelen_max == 1);
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
  template <uint32_t key_length>
  using index_type = cuco::static_map<key_type_t<key_length>, value_slice_type>;

  template <uint32_t key_length>
  index_type<key_length>* get_index() {
    return reinterpret_cast<index_type<key_length>*>(index_);
  }

  configs configs_;
  void* index_;
};
