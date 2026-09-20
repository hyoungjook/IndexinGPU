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
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <adapter_util.hpp>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <libcuckoo/cuckoohash_map.hh>

struct cpu_libcuckoo_adapter {
  static constexpr bool is_ordered = false;
  static constexpr bool support_mixed = true;
  static constexpr bool support_update = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  struct key_type {
    const key_slice_type* data;
    size_type length;
  };
  struct key_hash {
    static uint64_t mix64(uint64_t x) {
      x ^= x >> 30;
      x *= 0xbf58476d1ce4e5b9ULL;
      x ^= x >> 27;
      x *= 0x94d049bb133111ebULL;
      x ^= x >> 31;
      return x;
    }
    std::size_t operator()(const key_type& key) const {
      uint64_t hash = 1469598103934665603ULL;
      for (size_t i = 0; i < key.length; i++) {
        hash ^= key.data[i];
        hash *= 1099511628211ULL;
      }
      return static_cast<std::size_t>(mix64(hash ^ (static_cast<uint64_t>(key.length) << 32)));
    }
  };
  struct key_equal {
    bool operator()(const key_type& lhs, const key_type& rhs) const {
      if (lhs.length != rhs.length) {
        return false;
      }
      if (lhs.length == 0) {
        return true;
      }
      return std::memcmp(lhs.data, rhs.data, sizeof(key_slice_type) * lhs.length) == 0;
    }
  };
  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void register_dataset(const key_slice_type* keys,
                        const size_type* key_lengths,
                        const value_type* values,
                        const size_type* value_lengths) {
    (void)keys;
    (void)key_lengths;
    (void)values;
    (void)value_lengths;
  }
  void initialize() {
    dispatch_value_length([&](auto t) {
      index_ = new index_type<t.value>(configs_.initial_capacity);
    });
  }
  void destroy() {
    dispatch_value_length([&](auto t) {
      delete get_index<t.value>();
    });
    index_ = nullptr;
  }
  void thread_enter([[maybe_unused]] unsigned thread_idx) noexcept {}
  void thread_exit([[maybe_unused]] unsigned thread_idx) noexcept {}
  void insert(const key_slice_type* key,
              size_type key_length,
              const value_type* value,
              size_type value_length,
              std::size_t tuple_id,
              unsigned thread_idx) {
    (void)value_length;
    (void)tuple_id;
    (void)thread_idx;
    dispatch_value_length([&](auto t) {
      get_index<t.value>()->insert_or_assign(
        key_type{key, key_length}, make_value<t.value>(value));
    });
  }
  void update(const key_slice_type* key,
              size_type key_length,
              const value_type* value,
              size_type value_length,
              std::size_t tuple_id,
              unsigned thread_idx) {
    (void)value_length;
    (void)tuple_id;
    (void)thread_idx;
    dispatch_value_length([&](auto t) {
      get_index<t.value>()->update(
        key_type{key, key_length}, make_value<t.value>(value));
    });
  }
  void erase(const key_slice_type* key, size_type key_length, [[maybe_unused]] unsigned thread_idx) {
    dispatch_value_length([&](auto t) {
      get_index<t.value>()->erase(key_type{key, key_length});
    });
  }
  void find(const key_slice_type* key,
            size_type key_length,
            value_type* result,
            size_type* result_length,
            [[maybe_unused]] unsigned thread_idx) {
    dispatch_value_length([&](auto t) {
      stored_value<t.value> value;
      if (get_index<t.value>()->find(key_type{key, key_length}, value)) {
        std::memcpy(result, value.data, sizeof(value));
        *result_length = t.value;
      }
      else {
        result[0] = invalid_value;
        *result_length = 0;
      }
    });
  }
  void print_stats() {}
  void ht_print_load_factor(std::size_t max_keys, uint32_t key_length, uint32_t value_length) {
    (void)max_keys;
    (void)key_length;
    (void)value_length;
  }

 private:
  #define FORALL_ARGUMENTS_CPU_LIBCUCKOO(x) \
    x(initial_capacity, std::size_t, 1000000)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_CPU_LIBCUCKOO(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    uint32_t valuelen_max;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_CPU_LIBCUCKOO(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      #define PARSE_DEFAULT_ARGUMENTS(arg, type, default_value) \
      [[maybe_unused]] auto tmp_##arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS(PARSE_DEFAULT_ARGUMENTS)
      #undef PARSE_DEFAULT_ARGUMENTS
      valuelen_max = tmp_valuelen_max;
      check_argument(0 < initial_capacity);
      check_argument(tmp_valuelen_min == tmp_valuelen_max);
      check_argument(valuelen_max == 1 || valuelen_max == 2 ||
                     valuelen_max == 4 || valuelen_max == 8 ||
                     valuelen_max == 16);
      check_argument(valuelen_max == 1 ||
                     (tmp_keylen_min == 1 && tmp_keylen_max == 1));
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_CPU_LIBCUCKOO(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_CPU_LIBCUCKOO

  template <uint32_t value_length>
  struct stored_value {
    value_type data[value_length];
  };

  template <uint32_t value_length>
  using index_type = libcuckoo::cuckoohash_map<
    key_type, stored_value<value_length>, key_hash, key_equal>;

  template <typename function_type>
  void dispatch_value_length(function_type&& function) {
    adapter_util::dispatch_uint32<1, 2, 4, 8, 16>(
      configs_.valuelen_max, std::forward<function_type>(function));
  }

  template <uint32_t value_length>
  index_type<value_length>* get_index() {
    return reinterpret_cast<index_type<value_length>*>(index_);
  }

  template <uint32_t value_length>
  static stored_value<value_length> make_value(const value_type* value) {
    stored_value<value_length> result;
    std::memcpy(result.data, value, sizeof(result));
    return result;
  }

  configs configs_;
  void* index_ = nullptr;
};
