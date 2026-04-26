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

#include <stdlib.h>
#include <algorithm>
#include <execution>
#include <cstdint>
#include <numeric>
#include <vector>
#include <cmd.hpp>
#include <macros.hpp>
#include <generate_workload.hpp>
#if !defined(NOGPU)
#include <cuda_profiler_api.h>
#endif
#if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
#include <cpu_libcuckoo_adapter.hpp>
#include <cpu_masstree_adapter.hpp>
#include <cpu_art_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE)
#include <gpu_blink_tree_adapter.hpp>
#include <gpu_dycuckoo_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_MASSTREE)
#include <gpu_masstree_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_CHAINHASHTABLE)
#include <gpu_chainhashtable_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_CUCKOOHASHTABLE)
#include <gpu_cuckoohashtable_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_EXTENDHASHTABLE)
#include <gpu_extendhashtable_adapter.hpp>
#else
#include <gpu_masstree_adapter.hpp>
#include <gpu_chainhashtable_adapter.hpp>
#include <gpu_cuckoohashtable_adapter.hpp>
#include <gpu_extendhashtable_adapter.hpp>
#endif

namespace universal {

struct lap_timer {
  static float convert_s_to_Mops(float sec, std::size_t size) {
    return static_cast<float>(size) / 1e6 / sec;
  }
  float get_avg_Mops(std::size_t size) {
    float sum = 0;
    for (auto t: times_) sum += convert_s_to_Mops(t, size);
    return sum / times_.size();
  }
  void print_rate_Mops(std::string name, std::size_t size, bool print_all) {
    std::cout << name << ": " << get_avg_Mops(size) << " Mop/s";
    if (print_all) {
      std::cout << " (" << times_.size() << "; ";
      for (float t: times_) {
        std::cout << convert_s_to_Mops(t, size) << " ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;
  }
  std::vector<float> times_;
};

struct cpu_lap_timer: public lap_timer {
  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  void stop() {
    end_ = std::chrono::high_resolution_clock::now();
  }
  void record() {
    times_.push_back(std::chrono::duration_cast<std::chrono::duration<float>>(end_ - start_).count());
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> start_, end_;
};

#if !defined(NOGPU)
template <typename T>
struct device_vector {
  device_vector(std::size_t size, bool use_pinned_host_memory = false, bool nullify = false)
      : d_buffer_(nullptr), use_pinned_host_memory_(use_pinned_host_memory) {
    if (!nullify && size > 0) {
      if (use_pinned_host_memory) {
        cuda_try(cudaMallocHost(&d_buffer_, sizeof(T) * size));
      }
      else {
        cuda_try(cudaMalloc(&d_buffer_, sizeof(T) * size));
      }
    }
  }
  device_vector(const T* h_buffer, std::size_t size, bool use_pinned_host_memory = false, bool nullify = false)
      : device_vector(size, use_pinned_host_memory, nullify) {
    if (!nullify && size > 0) {
      if (use_pinned_host_memory) {
        std::copy(std::execution::par, h_buffer, h_buffer + size, d_buffer_);
      }
      else {
        cuda_try(cudaMemcpyAsync(d_buffer_, h_buffer, sizeof(T) * size, cudaMemcpyHostToDevice));
      }
    }
  }
  device_vector(std::vector<T>& h_vector, bool use_pinned_host_memory = false, bool nullify = false)
    : device_vector(h_vector.data(), h_vector.size(), use_pinned_host_memory, nullify) {}
  ~device_vector() {
    if (d_buffer_) {
      if (use_pinned_host_memory_) {
        cuda_try(cudaFreeHost(d_buffer_));
      }
      else {
        cuda_try(cudaFree(d_buffer_));
      }
    }
  }
  T* data() const { return d_buffer_; }
 private:
  T* d_buffer_;
  bool use_pinned_host_memory_;
};
#endif

#if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
static bool varlen_key_big_endian = true;
#else
static bool varlen_key_big_endian = false;
#endif

template <typename adapter_type>
struct bench_runner {
static void prefill(adapter_type& adapter,
                    std::vector<key_slice_type>& h_keys,
                    std::vector<size_type>& h_key_lengths,
                    std::vector<value_slice_type>& h_values,
                    std::vector<size_type>& h_value_lengths,
                    uint32_t keylen_min,
                    uint32_t keylen_max,
                    uint32_t valuelen_min,
                    uint32_t valuelen_max,
                    std::size_t num_keys) {
  adapter.initialize();
  #if !defined(NOGPU)
  static const uint32_t prefill_batch = 100 * 1000 * 1000;
  for (std::size_t begin = 0; begin < num_keys; begin += prefill_batch) {
    auto batch_size = std::min<std::size_t>(num_keys - begin, prefill_batch);
    auto d_keys = device_vector<key_slice_type>(&h_keys[begin * keylen_max], batch_size * keylen_max);
    auto d_key_lengths = device_vector<size_type>(&h_key_lengths[begin], batch_size, false, keylen_min == keylen_max);
    auto d_values = device_vector<value_slice_type>(&h_values[begin * valuelen_max], batch_size * valuelen_max);
    auto d_value_lengths = device_vector<size_type>(&h_value_lengths[begin], batch_size, false, valuelen_min == valuelen_max);
    adapter.insert(d_keys.data(), keylen_max, d_key_lengths.data(), d_values.data(), valuelen_max, d_value_lengths.data(), batch_size);
    cuda_try(cudaDeviceSynchronize());
  }
  #else
  helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
      adapter.insert(&h_keys[task_idx * keylen_max], h_key_lengths[task_idx], h_values[task_idx], task_idx, thread_id);
    }, num_keys,
    [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
    [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
  #endif
}

static void check_space(adapter_type& adapter,
                        std::vector<key_slice_type>& h_keys,
                        std::vector<size_type>& h_key_lengths,
                        std::vector<value_slice_type>& h_values,
                        std::vector<size_type>& h_value_lengths,
                        args_type& args) {
  prefill(adapter, h_keys, h_key_lengths, h_values, h_value_lengths, 
          args.keylen_min, args.keylen_max, args.valuelen_min, args.valuelen_max, args.max_keys);
  adapter.print_stats();
  adapter.destroy();
}

static void run_bench(adapter_type& adapter,
                      std::vector<key_slice_type>& h_keys,
                      std::vector<size_type>& h_key_lengths,
                      std::vector<value_slice_type>& h_values,
                      std::vector<size_type>& h_value_lengths,
                      std::vector<key_slice_type>& h_lookup_keys,
                      std::vector<size_type>& h_lookup_key_lengths,
                      std::vector<key_slice_type>& h_scan_upper_keys_if_btree,
                      std::vector<kernels::request_type> h_mix_types,
                      std::vector<key_slice_type> h_mix_keys,
                      std::vector<size_type> h_mix_key_lengths,
                      std::vector<value_slice_type> h_mix_values,
                      std::vector<size_type> h_mix_value_lengths,
                      [[maybe_unused]] std::vector<std::size_t> h_mix_key_tuple_ids,
                      args_type& args,
                      std::size_t result_buffer_size,
                      std::size_t result_length_buffer_size,
                      bool verbose,
                      bool print_all_measurements) {
  const bool use_null_keylength = (args.keylen_min == args.keylen_max);
  const bool use_null_valuelength = (args.valuelen_min == args.valuelen_max);
  // measure lookup & scan
  if (args.rep_lookup > 0 || args.rep_scan > 0) {
    cpu_lap_timer lookup_timer, scan_timer;
    prefill(adapter, h_keys, h_key_lengths, h_values, h_value_lengths, args.keylen_min, args.keylen_max, args.valuelen_min, args.valuelen_max, args.max_keys);
    #if !defined(NOGPU)
    auto d_lookup_keys = device_vector<key_slice_type>(h_lookup_keys, args.use_pinned_host_memory);
    auto d_lookup_key_lengths = device_vector<size_type>(h_lookup_key_lengths, args.use_pinned_host_memory, use_null_keylength);
    auto d_results = device_vector<value_slice_type>(result_buffer_size, args.use_pinned_host_memory);
    auto d_result_lengths = device_vector<size_type>(result_length_buffer_size, args.use_pinned_host_memory); // no null_valuelen b/c it's result
    auto d_scan_upper_keys_if_btree = device_vector<key_slice_type>(h_scan_upper_keys_if_btree, args.use_pinned_host_memory);
    cuda_try(cudaDeviceSynchronize());
    #else
    std::vector<value_slice_type> h_results(result_buffer_size);
    #endif
    for (uint32_t r = 0; r < args.rep_lookup; r++) {
      lookup_timer.start();
      #if !defined(NOGPU)
      adapter.find(d_lookup_keys.data(), args.keylen_max, d_lookup_key_lengths.data(), d_results.data(), args.valuelen_max, d_result_lengths.data(), args.num_lookups);
      #else
      helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
          h_results[task_idx] = adapter.find(&h_lookup_keys[task_idx * args.keylen_max], h_lookup_key_lengths[task_idx], thread_id);
        }, args.num_lookups,
        [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
        [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
      #endif
      #if !defined(NOGPU)
      cuda_try(cudaDeviceSynchronize());
      #endif
      lookup_timer.stop();
      lookup_timer.record();
      if (verbose) { std::cout << "lookup tested " << r + 1 << "/" << args.rep_lookup << std::endl; }
    }
    if constexpr (adapter_type::is_ordered) {
      for (uint32_t r = 0; r < args.rep_scan; r++) {
        scan_timer.start();
        #if !defined(NOGPU)
        adapter.scan(d_lookup_keys.data(), args.keylen_max, d_lookup_key_lengths.data(), args.scan_count, d_results.data(), args.valuelen_max, d_result_lengths.data(), args.num_scans, d_scan_upper_keys_if_btree.data());
        #else
        helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
            adapter.scan(&h_lookup_keys[task_idx * args.keylen_max], h_lookup_key_lengths[task_idx], args.scan_count, &h_results[task_idx * args.scan_count], thread_id);
          }, args.num_scans,
          [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
          [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
        #endif
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        scan_timer.stop();
        scan_timer.record();
        if (verbose) { std::cout << "scan tested " << r + 1 << "/" << args.rep_scan << std::endl; }
      }
    }
    adapter.destroy();
    if (args.rep_lookup > 0) {
      lookup_timer.print_rate_Mops("lookup", args.num_lookups, print_all_measurements);
    }
    if (adapter_type::is_ordered && args.rep_scan > 0) {
      scan_timer.print_rate_Mops("scan", args.num_scans, print_all_measurements);
    }
  }

  // measure insert & delete
  if (args.rep_insdel > 0) {
    cpu_lap_timer insert_timer, delete_timer;
    std::size_t num_prefill = args.max_keys - args.num_insdel;
    for (uint32_t r = 0; r < args.rep_insdel; r++) {
      prefill(adapter, h_keys, h_key_lengths, h_values, h_value_lengths, args.keylen_min, args.keylen_max, args.valuelen_min, args.valuelen_max, num_prefill);
      {
        #if !defined(NOGPU)
        auto d_insert_keys = device_vector<key_slice_type>(&h_keys[num_prefill * args.keylen_max], static_cast<std::size_t>(args.num_insdel) * args.keylen_max, args.use_pinned_host_memory);
        auto d_insert_key_lengths = device_vector<size_type>(&h_key_lengths[num_prefill], args.num_insdel, args.use_pinned_host_memory, use_null_keylength);
        auto d_insert_values = device_vector<value_slice_type>(&h_values[num_prefill * args.valuelen_max], static_cast<std::size_t>(args.num_insdel) * args.valuelen_max, args.use_pinned_host_memory);
        auto d_insert_value_lengths = device_vector<size_type>(&h_value_lengths[num_prefill], args.num_insdel, args.use_pinned_host_memory, use_null_valuelength);
        cuda_try(cudaDeviceSynchronize());
        #endif
        insert_timer.start();
        #if !defined(NOGPU)
        adapter.insert(d_insert_keys.data(), args.keylen_max, d_insert_key_lengths.data(), d_insert_values.data(), args.valuelen_max, d_insert_value_lengths.data(), args.num_insdel);
        #else
        helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
            auto tuple_id = num_prefill + task_idx;
            adapter.insert(&h_keys[tuple_id * args.keylen_max],
                           h_key_lengths[tuple_id],
                           h_values[tuple_id],
                           tuple_id,
                           thread_id);
          }, args.num_insdel,
          [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
          [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
        #endif
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        insert_timer.stop();
        insert_timer.record();
      }
      {
        #if !defined(NOGPU)
        auto d_delete_keys = device_vector<key_slice_type>(&h_keys[0], static_cast<std::size_t>(args.num_insdel) * args.keylen_max, args.use_pinned_host_memory);
        auto d_delete_key_lengths = device_vector<size_type>(&h_key_lengths[0], args.num_insdel, args.use_pinned_host_memory, use_null_keylength);
        cuda_try(cudaDeviceSynchronize());
        #endif
        delete_timer.start();
        #if !defined(NOGPU)
        adapter.erase(d_delete_keys.data(), args.keylen_max, d_delete_key_lengths.data(), args.num_insdel);
        #else
        helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
            adapter.erase(&h_keys[task_idx * args.keylen_max],
                          h_key_lengths[task_idx],
                          thread_id);
          }, args.num_insdel,
          [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
          [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
        #endif
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        delete_timer.stop();
        delete_timer.record();
      }
      adapter.destroy();
      if (verbose) { std::cout << "insert/delete tested " << r + 1 << "/" << args.rep_insdel << std::endl; }
    }
    insert_timer.print_rate_Mops("insert", args.num_insdel, print_all_measurements);
    delete_timer.print_rate_Mops("delete", args.num_insdel, print_all_measurements);
  }

  // space test
  #if !defined(NOGPU)
  if (args.rep_space > 0) {
    uint32_t rep_del = args.max_keys / args.num_space;
    // repeat just once, ignore rep_space, as we don't measure time here
    for (uint32_t r = 0; r < 1; r++) {
      prefill(adapter, h_keys, h_key_lengths, h_values, h_value_lengths, args.keylen_min, args.keylen_max, args.valuelen_min, args.valuelen_max, args.max_keys);
      if (r == 0) { adapter.print_stats(); }
      for (uint32_t d = 0; d < rep_del; d++) {
        auto d_delete_keys = device_vector<key_slice_type>(
          &h_keys[static_cast<std::size_t>(args.num_space) * args.keylen_max * d],
          static_cast<std::size_t>(args.num_space) * args.keylen_max);
        auto d_delete_key_lengths = device_vector<size_type>(
          &h_key_lengths[static_cast<std::size_t>(args.num_space) * d], args.num_space, false, use_null_keylength);
        adapter.erase(d_delete_keys.data(), args.keylen_max, d_delete_key_lengths.data(), args.num_space);
        cuda_try(cudaDeviceSynchronize());
        if (r == 0) { adapter.print_stats(); }
      }
      adapter.destroy();
      if (verbose) { std::cout << "space tested " << r + 1 << "/" << args.rep_space << std::endl; }
    }
  }
  #endif

  // measure mixed
  if constexpr (adapter_type::support_mixed) {
    if (args.rep_mixed > 0) {
      std::size_t num_prefill = args.max_keys - mix_get_num_insdel(args.num_mixed, args.mix_read_ratio);
      cpu_lap_timer mix_timer;
      for (uint32_t r = 0; r < args.rep_mixed; r++) {
        prefill(adapter, h_keys, h_key_lengths, h_values, h_value_lengths, args.keylen_min, args.keylen_max, args.valuelen_min, args.valuelen_max, num_prefill);
        #if !defined(NOGPU)
        auto d_mix_types = device_vector<kernels::request_type>(h_mix_types, args.use_pinned_host_memory);
        auto d_mix_keys = device_vector<key_slice_type>(h_mix_keys, args.use_pinned_host_memory);
        auto d_mix_key_lengths = device_vector<size_type>(h_mix_key_lengths, args.use_pinned_host_memory, use_null_keylength);
        auto d_mix_values = device_vector<value_slice_type>(h_mix_values, args.use_pinned_host_memory);
        auto d_mix_value_lengths = device_vector<size_type>(h_mix_value_lengths, args.use_pinned_host_memory); // no null, might contain results
        cuda_try(cudaDeviceSynchronize());
        #endif
        mix_timer.start();
        #if !defined(NOGPU)
        adapter.mixed_batch(d_mix_types.data(), d_mix_keys.data(), args.keylen_max, d_mix_key_lengths.data(), d_mix_values.data(), args.valuelen_max, d_mix_value_lengths.data(), args.num_mixed);
        #else
        helper_multithread([&](std::size_t task_idx, unsigned thread_id) {
            if (h_mix_types[task_idx] == kernels::request_type_find) {
              h_mix_values[task_idx] = adapter.find(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx], thread_id);
            }
            else if (h_mix_types[task_idx] == kernels::request_type_insert) {
              adapter.insert(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx], h_mix_values[task_idx], h_mix_key_tuple_ids[task_idx], thread_id);
            }
            else {
              adapter.erase(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx], thread_id);
            }
          }, args.num_mixed,
          [&](unsigned thread_id) { adapter.thread_enter(thread_id); },
          [&](unsigned thread_id) { adapter.thread_exit(thread_id); });
        #endif
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        mix_timer.stop();
        mix_timer.record();
        adapter.destroy();
        if (verbose) { std::cout << "mix tested " << r + 1 << "/" << args.rep_mixed << std::endl; }
      }
      mix_timer.print_rate_Mops("mixed", args.num_mixed, print_all_measurements);
    }
  }
}
};

} // namespace universal

#define UNIVERSAL_BENCH_DECLARE_TEMPLATES(adapter_type) \
  extern template struct bench_runner<adapter_type>;

#define UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES(adapter_type) \
  template struct bench_runner<adapter_type>;

#if !defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE) && \
    !defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE) && \
    !defined(UNIVERSAL_BENCH_SKIP_MAIN)
namespace universal {
UNIVERSAL_BENCH_DECLARE_TEMPLATES(gpu_masstree_adapter)
UNIVERSAL_BENCH_DECLARE_TEMPLATES(gpu_chainhashtable_adapter)
UNIVERSAL_BENCH_DECLARE_TEMPLATES(gpu_cuckoohashtable_adapter)
UNIVERSAL_BENCH_DECLARE_TEMPLATES(gpu_extendhashtable_adapter)
} // namespace universal
#endif

#if defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_MASSTREE)
namespace universal {
UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES(gpu_masstree_adapter)
} // namespace universal
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_CHAINHASHTABLE)
namespace universal {
UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES(gpu_chainhashtable_adapter)
} // namespace universal
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_CUCKOOHASHTABLE)
namespace universal {
UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES(gpu_cuckoohashtable_adapter)
} // namespace universal
#elif defined(UNIVERSAL_BENCH_INSTANTIATE_GPU_EXTENDHASHTABLE)
namespace universal {
UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES(gpu_extendhashtable_adapter)
} // namespace universal
#endif

#undef UNIVERSAL_BENCH_INSTANTIATE_TEMPLATES
#undef UNIVERSAL_BENCH_DECLARE_TEMPLATES

#if !defined(UNIVERSAL_BENCH_SKIP_MAIN)
int main(int argc, char** argv) {
  auto arg_strings = std::vector<std::string>(argv, argv + argc);
  universal::args_type args(arg_strings);
  bool verbose = get_arg_value<bool>(arg_strings, "verbose").value_or(false);
  bool print_all_measurements = get_arg_value<bool>(arg_strings, "print_all_measurements").value_or(false);

  #if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
  #define FORALL_INDEXES(x) \
    x(cpu_libcuckoo) x(cpu_masstree) x(cpu_art)
  #elif defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE)
  #define FORALL_INDEXES(x) \
    x(gpu_blink_tree) x(gpu_dycuckoo)
  #else
  #define FORALL_INDEXES(x) \
    x(gpu_masstree) x(gpu_chainhashtable) \
    x(gpu_cuckoohashtable) x(gpu_extendhashtable)
  #endif

  #define INDEX_NAME_CHECK(index) (args.index_type == #index) ||
  check_argument(FORALL_INDEXES(INDEX_NAME_CHECK) false);
  #undef INDEX_NAME_CHECK
  #define ADAPTER_DECLARE(index) index##_adapter index##_adapter_;
  FORALL_INDEXES(ADAPTER_DECLARE)
  #undef ADAPTER_DECLARE
  #define ADAPTER_PARSE_ARGS(index) \
  if (args.index_type == #index) { index##_adapter_.parse(arg_strings); }
  FORALL_INDEXES(ADAPTER_PARSE_ARGS)
  #undef ADAPTER_PARSE_ARGS

  // print arguments
  args.print();
  #define ADAPTER_PRINT_ARGS(index) \
  if (args.index_type == #index) { index##_adapter_.print_args(); }
  FORALL_INDEXES(ADAPTER_PRINT_ARGS)
  #undef ADAPTER_PRINT_ARGS
  check_argument(args.num_insdel <= args.max_keys &&
                 args.num_mixed <= args.max_keys &&
                 args.num_space <= args.max_keys);
  if (!args.only_check_space) {
    check_argument(args.rep_lookup > 0 || args.rep_scan > 0 ||
                   args.rep_insdel > 0 || args.rep_mixed > 0 ||
                   args.rep_space > 0);
    if (args.rep_space > 0) {
      check_argument(args.max_keys % args.num_space == 0);
    }
  }
  #if defined(NOGPU)
  // We only implemented 4B value for CPU indexes
  check_argument(args.valuelen_min == 1 && args.valuelen_max == 1);
  #endif

  // generate keys and queries
  if (verbose) { std::cout << "Generating workload..." << std::endl; }
  std::vector<key_slice_type> h_keys;
  std::vector<size_type> h_key_lengths;
  std::vector<value_slice_type> h_values;
  std::vector<size_type> h_value_lengths;
  std::vector<key_slice_type> h_lookup_keys;
  std::vector<size_type> h_lookup_key_lengths;
  std::vector<key_slice_type> h_scan_upper_keys_if_btree;
  std::vector<kernels::request_type> h_mix_types;
  std::vector<key_slice_type> h_mix_keys;
  std::vector<size_type> h_mix_key_lengths;
  std::vector<value_slice_type> h_mix_values;
  std::vector<size_type> h_mix_value_lengths;
  std::vector<std::size_t> h_mix_key_tuple_ids;
  // 1. generate keys: max_keys
  universal::generate_key_values(
    h_keys, h_key_lengths, h_values, h_value_lengths,
    args.max_keys, args.keylen_prefix, args.keylen_min, args.keylen_max, args.keylen_theta,
    args.valuelen_min, args.valuelen_max, args.valuelen_theta,
    universal::varlen_key_big_endian);
  // 2. generate lookup keys: max(num_lookups, num_scans)
  std::size_t num_lookups_keys = std::max<std::size_t>(
    (args.rep_lookup > 0) ? args.num_lookups : 0,
    (args.rep_scan > 0) ? args.num_scans : 0);
  if (num_lookups_keys > 0) {
    universal::generate_lookup_keys(
      h_lookup_keys, h_lookup_key_lengths, h_keys, h_key_lengths,
      args.max_keys, args.keylen_max, num_lookups_keys, args.lookup_theta);
  }
  if (args.index_type == "gpu_blink_tree" && args.rep_scan > 0) {
    check_argument(args.keylen_max == 1);
    h_scan_upper_keys_if_btree = std::vector<key_slice_type>(args.num_scans);
    for (std::size_t i = 0; i < args.num_scans; i++) {
      h_scan_upper_keys_if_btree[i] = h_lookup_keys[i] + args.scan_count - 1;
    }
  }
  // 3. result buffer: 1 per lookup, count per scan
  std::size_t scan_result_buffer_multiplier =
    (args.index_type == "gpu_blink_tree" || args.index_type == "cpu_art") ? 2 : 1;
  std::size_t result_buffer_size = std::max<std::size_t>(
    (args.rep_lookup > 0) ? (args.num_lookups * args.valuelen_max) : 0,
    (args.rep_scan > 0) ? (scan_result_buffer_multiplier * args.num_scans * args.scan_count * args.valuelen_max) : 0);
  std::size_t result_length_buffer_size = std::max<std::size_t>(
    (args.rep_lookup > 0) ? args.num_lookups : 0,
    (args.rep_scan > 0) ? (scan_result_buffer_multiplier * args.num_scans * args.scan_count) : 0);
  // 4. generate mixed keys
  if (args.rep_mixed > 0) {
    universal::generate_mixed_keys(
      h_mix_types, h_mix_keys, h_mix_key_lengths, h_mix_values, h_mix_value_lengths, h_mix_key_tuple_ids, h_keys, h_key_lengths, h_values, h_value_lengths,
      args.max_keys, args.keylen_max, args.valuelen_max, args.num_mixed, args.mix_read_ratio, args.mix_presort, args.lookup_theta);
  }

  // only check space?
  if (args.only_check_space) {
    if (verbose) { std::cout << "Checking space consumption..." << std::endl; }
    #define ADAPTER_CHECK_SPACE(index) \
    if (args.index_type == #index) { \
      universal::bench_runner<index##_adapter>::check_space(index##_adapter_, h_keys, h_key_lengths, h_values, h_value_lengths, args); \
    }
    FORALL_INDEXES(ADAPTER_CHECK_SPACE)
    #undef ADAPTER_CHECK_SPACE
    return 0;
  }

  // run benchmark
  if (verbose) { std::cout << "Running benchmark..." << std::endl; }
  #define ADAPTER_REGISTER_DATASET(index) \
  if (args.index_type == #index) { \
    index##_adapter_.register_dataset(h_keys.data(), h_key_lengths.data(), h_values.data()); \
  }
  #if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
  FORALL_INDEXES(ADAPTER_REGISTER_DATASET)
  #endif
  #undef ADAPTER_REGISTER_DATASET
  #define ADAPTER_RUN_BENCH(index) \
  if (args.index_type == #index) { \
    universal::bench_runner<index##_adapter>::run_bench(index##_adapter_, \
      h_keys, h_key_lengths, h_values, h_value_lengths, h_lookup_keys, h_lookup_key_lengths, h_scan_upper_keys_if_btree, \
      h_mix_types, h_mix_keys, h_mix_key_lengths, h_mix_values, h_mix_value_lengths, h_mix_key_tuple_ids, args, \
      result_buffer_size, result_length_buffer_size, verbose, print_all_measurements); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH
}
#endif
