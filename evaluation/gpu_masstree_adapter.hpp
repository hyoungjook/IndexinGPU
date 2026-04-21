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
#include <gpu_masstree.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

struct gpu_masstree_adapter {
  static constexpr bool is_ordered = true;
  static constexpr bool support_mixed = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using allocator_type = simple_slab_allocator<128>;
  using reclaimer_type = simple_debra_reclaimer<>;
  using index32_type = GpuMasstree::gpu_masstree<allocator_type, reclaimer_type, 32>;
  using index16_type = GpuMasstree::gpu_masstree<allocator_type, reclaimer_type, 16>;

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
      index_ = reinterpret_cast<void*>(new index32_type(*allocator_, *reclaimer_));
    }
    else {
      index_ = reinterpret_cast<void*>(new index16_type(*allocator_, *reclaimer_));
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
      adapter_util::dispatch_bool(configs_.enable_suffix, [&](auto t2, auto s2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto s3, auto r3) {
          adapter_util::dispatch_bool(configs_.use_shmem_key, [&](auto t4, auto s4, auto r4, auto k4) {
            do_insert<t4.value, s4.value, r4.value, k4.value>(keys, keylen_max, key_lengths, values, num_keys);
          }, t3, s3, r3);
        }, t2, s2);
      }, t1);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_uint32<0, 1, 2, 3, 4>(configs_.merge_level, [&](auto t2, auto m2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto m3, auto r3) {
          adapter_util::dispatch_bool(configs_.use_shmem_key, [&](auto t4, auto m4, auto r4, auto k4) {
            do_erase<t4.value, m4.value, r4.value, k4.value>(keys, keylen_max, key_lengths, num_keys);
          }, t3, m3, r3);
        }, t2, m2);
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
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto c3, auto r3) {
          adapter_util::dispatch_bool(configs_.use_shmem_key, [&](auto t4, auto c4, auto r4, auto k4) {
            do_find<t4.value, c4.value, r4.value, k4.value>(keys, keylen_max, key_lengths, results, num_keys);
          }, t3, c3, r3);
        }, t2, c2);
      }, t1);
    });
  }
  void scan(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            uint32_t count,
            value_type* results,
            std::size_t num_keys,
            const key_slice_type* upper_keys_for_btree) {
    (void)upper_keys_for_btree;
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.lookup_concurrent, [&](auto t2, auto c2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto c3, auto r3) {
          adapter_util::dispatch_bool(configs_.use_shmem_key, [&](auto t4, auto c4, auto r4, auto k4) {
            do_scan<t4.value, c4.value, r4.value, k4.value>(keys, key_lengths, keylen_max, count, num_keys,
                                                            nullptr, nullptr, nullptr, results, nullptr, nullptr);
          }, t3, c3, r3);
        }, t2, c2);
      }, t1);
    });
  }
  void mixed_batch(const kernels::request_type* types,
                   const key_slice_type* keys,
                   uint32_t keylen_max,
                   const size_type* key_lengths,
                   value_type* values,
                   std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.enable_suffix, [&](auto t2, auto s2) {
        adapter_util::dispatch_uint32<0, 1, 2, 3, 4>(configs_.merge_level, [&](auto t3, auto s3, auto m3) {
          adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t4, auto s4, auto m4, auto r4) {
            adapter_util::dispatch_bool(configs_.use_shmem_key, [&](auto t5, auto s5, auto m5, auto r5, auto k5) {
              do_mixed<t5.value, s5.value, m5.value, r5.value, k5.value>(types, keys, keylen_max, key_lengths, values, nullptr, num_keys);
            }, t4, s4, m4, r4);
          }, t3, s3, m3);
        }, t2, s2);
      }, t1);
    });
  }
  void print_stats() {
    allocator_->print_stats();
    //if (configs_.tile_size == 32) {
    //  reinterpret_cast<index32_type*>(index_)->validate();
    //}
    //else {
    //  reinterpret_cast<index16_type*>(index_)->validate();
    //}
  }

 private:
  #define FORALL_ARGUMENTS_GPU_MASSTREE(x) \
    x(allocator_pool_ratio, float, 0.9f) \
    x(tile_size, uint32_t, 16) \
    x(lookup_concurrent, bool, false) \
    x(enable_suffix, bool, true) \
    /* merge_level: 0(naive) 1(concurrent) 2(merge) 3(pessimistic_merge) 4(remove_root) */ \
    x(merge_level, uint32_t, 4) \
    x(reuse_root, bool, true) \
    x(use_shmem_key, bool ,false)
  struct configs {
    #define DECLARE_ARGUMENTS(arg, type, default_value) type arg;
    FORALL_ARGUMENTS_GPU_MASSTREE(DECLARE_ARGUMENTS)
    #undef DECLARE_ARGUMENTS
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_ARGUMENTS(arg, type, default_value) \
      arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS_GPU_MASSTREE(PARSE_ARGUMENTS)
      #undef PARSE_ARGUMENTS
      check_argument(tile_size == 32 || tile_size == 16);
      check_argument(merge_level <= 4);
    }
    void print() const {
      #define PRINT_ARGUMENTS(arg, type, default_value) \
      std::cout << "    " #arg "=" << arg << std::endl;
      FORALL_ARGUMENTS_GPU_MASSTREE(PRINT_ARGUMENTS)
      #undef PRINT_ARGUMENTS
    }
  };
  #undef FORALL_ARGUMENTS_GPU_MASSTREE

  template <uint32_t tile_size, bool enable_suffix, bool reuse_root, bool use_shmem_key, typename... arg_types>
  void do_insert(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template insert<enable_suffix, reuse_root, use_shmem_key>(args...);
  }

  template <uint32_t tile_size, uint32_t merge_level, bool reuse_root, bool use_shmem_key, typename... arg_types>
  void do_erase(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template erase<merge_level >= 4, merge_level >= 3, merge_level >= 2, merge_level >= 1, reuse_root, use_shmem_key>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, bool reuse_root, bool use_shmem_key, typename... arg_types>
  void do_find(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template find<lookup_concurrent, reuse_root, use_shmem_key>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, bool reuse_root, bool use_shmem_key, typename... arg_types>
  void do_scan(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template scan<false, lookup_concurrent, reuse_root, use_shmem_key>(args...);
  }

  template <uint32_t tile_size, bool enable_suffix, uint32_t merge_level, bool reuse_root, bool use_shmem_key, typename... arg_types>
  void do_mixed(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template mixed_batch<enable_suffix, merge_level >= 4, merge_level >= 3, merge_level >= 2, reuse_root, use_shmem_key>(args...);
  }

  configs configs_;
  allocator_type* allocator_;
  reclaimer_type* reclaimer_;
  void* index_;
};
