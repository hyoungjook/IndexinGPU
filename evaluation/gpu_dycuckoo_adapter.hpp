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
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <gpu_dycuckoo_backend.hpp>
#include <macros.hpp>

struct gpu_dycuckoo_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    if (configs_.use_lock) {
      index_ = gpu_dycuckoo_dynamic_lock_create(configs_.initial_capacity,
                                                configs_.small_batch_size,
                                                configs_.fill_factor_lower_bound,
                                                configs_.fill_factor_upper_bound);
    }
    else {
      index_ = gpu_dycuckoo_dynamic_create(configs_.initial_capacity,
                                           configs_.small_batch_size,
                                           configs_.fill_factor_lower_bound,
                                           configs_.fill_factor_upper_bound);
    }
  }
  void destroy() {
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_destroy(index_);
    }
    else {
      gpu_dycuckoo_dynamic_destroy(index_);
    }
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_insert(index_, keys, values, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_insert(index_, keys, values, num_keys);
    }
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_erase(index_, keys, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_erase(index_, keys, num_keys);
    }
  }
  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_find(index_, keys, results, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_find(index_, keys, results, num_keys);
    }
  }

 private:
  #define FORALL_ARGUMENTS_GPU_DYCUCKOO(x) \
    x(use_lock, bool, false) \
    x(initial_capacity, uint32_t, 1000000) \
    x(fill_factor_lower_bound, float, 0.5f) \
    x(fill_factor_upper_bound, float, 0.8f) \
    x(small_batch_size, int, 20000)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_GPU_DYCUCKOO(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_GPU_DYCUCKOO(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      check_argument(0 < initial_capacity);
      check_argument(0 < fill_factor_lower_bound &&
                     fill_factor_lower_bound < fill_factor_upper_bound &&
                     fill_factor_upper_bound <= 1.0f);
      #define PARSE_DEFAULT_ARGUMENTS(arg, type, default_value) \
      [[maybe_unused]] auto arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS(PARSE_DEFAULT_ARGUMENTS)
      #undef PARSE_DEFAULT_ARGUMENTS
      check_argument(keylen_max == 1);
    }

    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_GPU_DYCUCKOO(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_GPU_DYCUCKOO

  configs configs_;
  void* index_;
};
