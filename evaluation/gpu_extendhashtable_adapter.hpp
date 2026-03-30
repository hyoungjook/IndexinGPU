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
#include <vector>
#include <string>
#include <macros.hpp>
#include <cmd.hpp>
#include <adapter_util.hpp>
#include <gpu_extendhashtable.hpp>
#include <simple_slab_linear_alloc.hpp>
#include <simple_debra_reclaim.hpp>

struct gpu_extendhashtable_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using allocator_type = simple_slab_linear_allocator<128>;
  using reclaimer_type = simple_debra_reclaimer<>;
  using index32_type = GpuExtendHashtable::gpu_extendhashtable<allocator_type, reclaimer_type, 32>;
  using index16_type = GpuExtendHashtable::gpu_extendhashtable<allocator_type, reclaimer_type, 16>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    allocator_ = new allocator_type(configs_.allocator_pool_ratio);
    reclaimer_ = new reclaimer_type();
    if (configs_.tile_size == 32) {
      index_ = reinterpret_cast<void*>(new index32_type(*allocator_, *reclaimer_,
        configs_.initial_directory_size, configs_.resize_policy, configs_.load_factor_threshold));
    }
    else {
      index_ = reinterpret_cast<void*>(new index16_type(*allocator_, *reclaimer_,
        configs_.initial_directory_size, configs_.resize_policy, configs_.load_factor_threshold));
    }
  }
  void destroy() {
    if (configs_.tile_size == 32) {
      delete reinterpret_cast<index32_type*>(index_);
    }
    else {
      delete reinterpret_cast<index16_type*>(index_);
    }
    delete allocator_;
    delete reclaimer_;
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t2, auto h2) {
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.merge_level, [&](auto t3, auto h3, auto m3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto h4, auto m4, auto r4) {
            do_insert<t4.value, h4.value, m4.value, r4.value>(keys, keylen_max, key_lengths, values, num_keys);
          }, t3, h3, m3);
        }, t2, h2);
      }, t1);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t2, auto h2) {
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.merge_level, [&](auto t3, auto h3, auto m3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto h4, auto m4, auto r4) {
            do_erase<t4.value, h4.value, m4.value, r4.value>(keys, keylen_max, key_lengths, num_keys);
          }, t3, h3, m3);
        }, t2, h2);
      }, t1);
    });
  }
  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.lookup_concurrent, [&](auto t2, auto c2) {
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t3, auto c3, auto h3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto c4, auto h4, auto r4) {
            do_find<t4.value, c4.value, h4.value, r4.value>(keys, keylen_max, key_lengths, results, num_keys);
          }, t3, c3, h3);
        }, t2, c2);
      }, t1);
    });
  }

 private:
  #define FORALL_ARGUMENTS_GPU_EXTENDHASHTABLE(x) \
    x(allocator_pool_ratio, float, 0.9f) \
    x(tile_size, uint32_t, 32) \
    x(lookup_concurrent, bool, true) \
    x(initial_directory_size, uint32_t, (1000000 / 15)) \
    x(resize_policy, float, 2.0f) \
    x(load_factor_threshold, float, 2.5f) \
    /* hash_tag_level: 0(slice0_tag), 1(hash_tag), 2(samehash_tag) */ \
    x(hash_tag_level, uint32_t, 2) \
    /* merge_level: 0(naive), 1(merge_chains), 2(merge_buckets) */ \
    x(merge_level, uint32_t, 2) \
    x(reuse_dirsize, bool, true)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_GPU_EXTENDHASHTABLE(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_GPU_EXTENDHASHTABLE(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      check_argument(tile_size == 32 || tile_size == 16);
      check_argument(0 < load_factor_threshold);
      check_argument(hash_tag_level <= 2);
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_GPU_EXTENDHASHTABLE(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_GPU_EXTENDHASHTABLE

  template <uint32_t tile_size, uint32_t hash_tag_level, uint32_t merge_level, bool reuse_dirsize, typename... arg_types>
  void do_insert(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template insert<hash_tag_level >= 1, hash_tag_level >= 2, merge_level >= 1, reuse_dirsize>(args...);
  }

  template <uint32_t tile_size, uint32_t hash_tag_level, uint32_t merge_level, bool reuse_dirsize, typename... arg_types>
  void do_erase(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template erase<hash_tag_level >= 1, hash_tag_level >= 2, merge_level >= 2, merge_level >= 1, reuse_dirsize>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, uint32_t hash_tag_level, bool reuse_dirsize, typename... arg_types>
  void do_find(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template find<lookup_concurrent, hash_tag_level >= 1, hash_tag_level >= 2, reuse_dirsize>(args...);
  }

  configs configs_;
  allocator_type* allocator_;
  reclaimer_type* reclaimer_;
  void* index_;
};
