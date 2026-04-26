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

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cmd.hpp>
#include <tbb/concurrent_hash_map.h>

struct cpu_onetbb_adapter {
  static constexpr bool is_ordered = false;
  static constexpr bool support_mixed = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  struct key_type {
    const key_slice_type* data;
    size_type length;
  };

  struct key_compare {
    static uint64_t mix64(uint64_t x) {
      x ^= x >> 30;
      x *= 0xbf58476d1ce4e5b9ULL;
      x ^= x >> 27;
      x *= 0x94d049bb133111ebULL;
      x ^= x >> 31;
      return x;
    }

    std::size_t hash(const key_type& key) const {
      uint64_t hash = 1469598103934665603ULL;
      for (std::size_t i = 0; i < key.length; i++) {
        hash ^= key.data[i];
        hash *= 1099511628211ULL;
      }
      return static_cast<std::size_t>(mix64(hash ^ (static_cast<uint64_t>(key.length) << 32)));
    }

    bool equal(const key_type& lhs, const key_type& rhs) const {
      if (lhs.length != rhs.length) {
        return false;
      }
      if (lhs.length == 0) {
        return true;
      }
      return std::memcmp(lhs.data, rhs.data, sizeof(key_slice_type) * lhs.length) == 0;
    }
  };

  using allocator_type = std::allocator<std::pair<const key_type, value_type>>;
  using index_type = tbb::concurrent_hash_map<key_type, value_type, key_compare, allocator_type>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }

  void print_args() const {
    configs_.print();
  }

  void register_dataset(const key_slice_type* key, const size_type* key_lengths, const value_type* values) {}

  void initialize() {
    index_ = std::make_unique<index_type>(configs_.initial_capacity, key_compare{});
  }

  void destroy() {
    index_.reset();
  }

  void thread_enter([[maybe_unused]] unsigned thread_idx) noexcept {}

  void thread_exit([[maybe_unused]] unsigned thread_idx) noexcept {}

  void insert(const key_slice_type* key, size_type key_length, value_type value, std::size_t tuple_id, unsigned thread_idx) {
    (void)tuple_id;
    (void)thread_idx;
    typename index_type::accessor accessor;
    index_->insert(accessor, key_type{key, key_length});
    accessor->second = value;
  }

  void erase(const key_slice_type* key, size_type key_length, [[maybe_unused]] unsigned thread_idx) {
    index_->erase(key_type{key, key_length});
  }

  value_type find(const key_slice_type* key, size_type key_length, [[maybe_unused]] unsigned thread_idx) {
    typename index_type::const_accessor accessor;
    if (index_->find(accessor, key_type{key, key_length})) {
      return accessor->second;
    }
    return invalid_value;
  }

  void print_stats() {}

 private:
  #define FORALL_ARGUMENTS_CPU_ONETBB(x) \
    x(initial_capacity, std::size_t, 1000000)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_CPU_ONETBB(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_CPU_ONETBB(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      check_argument(0 < initial_capacity);
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_CPU_ONETBB(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_CPU_ONETBB

  configs configs_;
  std::unique_ptr<index_type> index_;
};
