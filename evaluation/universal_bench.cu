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
#include <cstdint>
#include <numeric>
#include <vector>
#include <cmd.hpp>
#include <macros.hpp>
#include <generate_workload.hpp>
#if !defined(NOGPU)
#include <cuda_profiler_api.h>
#include <thrust/sequence.h>
#include <gpu_timer.hpp>
#endif
#if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
#include <cpu_libcuckoo_adapter.hpp>
#include <cpu_masstree_adapter.hpp>
#include <cpu_art_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE)
#include <gpu_blink_tree_adapter.hpp>
#include <gpu_dycuckoo_adapter.hpp>
#else
#include <gpu_masstree_adapter.hpp>
#include <gpu_chainhashtable_adapter.hpp>
#include <gpu_cuckoohashtable_adapter.hpp>
#include <gpu_extendhashtable_adapter.hpp>
#endif

namespace universal {

struct lap_timer {
  float get_avg() {
    float sum = 0;
    for (auto t: times_) sum += t;
    return sum / times_.size();
  }
  float get_max() {
    float mx = times_[0];
    for (auto t: times_) mx = std::max(mx, t);
    return mx;
  }
  float get_min() {
    float mn = times_[0];
    for (auto t: times_) mn = std::min(mn, t);
    return mn;
  }
  void print_rate_Mops(std::string name, std::size_t size) {
    auto min_rate = static_cast<float>(size) / 1e6 / get_max();
    auto avg_rate = static_cast<float>(size) / 1e6 / get_avg();
    auto max_rate = static_cast<float>(size) / 1e6 / get_min();
    std::cout << name << ": " << avg_rate << " Mop/s (" << min_rate << ", " << max_rate << ")" << std::endl;
  }
  std::vector<float> times_;
};

#if !defined(NOGPU)
struct gpu_lap_timer: public lap_timer {
  void start() {
    timer_.start_timer();
  }
  void stop() {
    timer_.stop_timer();
  }
  void record() {
    times_.push_back(timer_.get_elapsed_s());
  }
  gpu_timer timer_;
};
#endif

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

#if defined(UNIVERSAL_BENCH_WITH_CPU_BASELINE)
using timer_type = cpu_lap_timer;
#else
using timer_type = gpu_lap_timer;
#endif

template <class F, class ThreadEnter, class ThreadExit>
void helper_multithread(F&& f, std::size_t num_tasks, ThreadEnter&& thread_enter, ThreadExit&& thread_exit) {
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    workers.emplace_back([&](unsigned thread_id) {
      thread_enter();
      for (std::size_t task_idx = thread_id; task_idx < num_tasks; task_idx += num_workers) {
        std::forward<F>(f)(task_idx);
      }
      thread_exit();
    }, tid);
  }
  for (auto& w: workers) { w.join(); }
}

template <typename adapter_type>
void prefill(adapter_type& adapter,
             std::vector<key_slice_type>& h_keys,
             std::vector<size_type>& h_key_lengths,
             std::vector<value_type>& h_values,
             uint32_t keylen_max,
             std::size_t num_prefill) {
  adapter.initialize();
  #if !defined(NOGPU)
  static const uint32_t prefill_batch = 16 * 1024 * 1024;
  for (std::size_t begin = 0; begin < num_prefill; begin += prefill_batch) {
    auto batch_size = std::min<std::size_t>(num_prefill - begin, prefill_batch);
    auto d_keys = thrust::device_vector<key_slice_type>(
      h_keys.begin() + (keylen_max * begin), h_keys.begin() + (keylen_max * (begin + batch_size)));
    auto d_key_lengths = thrust::device_vector<size_type>(
      h_key_lengths.begin() + begin, h_key_lengths.begin() + (begin + batch_size));
    auto d_values = thrust::device_vector<value_type>(
      h_values.begin() + begin, h_values.begin() + (begin + batch_size));
    adapter.insert(d_keys.data().get(), keylen_max, d_key_lengths.data().get(), d_values.data().get(), batch_size);
    cuda_try(cudaDeviceSynchronize());
  }
  #else
  helper_multithread([&](std::size_t task_idx) {
      adapter.insert(&h_keys[task_idx * keylen_max], h_key_lengths[task_idx], h_values[task_idx]);
    }, num_prefill,
    [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
  #endif
}

template <typename adapter_type>
void run_bench(adapter_type& adapter,
               std::vector<key_slice_type>& h_keys,
               std::vector<size_type>& h_key_lengths,
               std::vector<value_type>& h_values,
               std::vector<key_slice_type>& h_lookup_keys,
               std::vector<size_type>& h_lookup_key_lengths,
               std::vector<key_slice_type>& h_scan_upper_keys_if_btree,
               std::vector<kernels::request_type> h_mix_types,
               std::vector<key_slice_type> h_mix_keys,
               std::vector<size_type> h_mix_key_lengths,
               std::vector<value_type> h_mix_values,
               args_type& args,
               std::size_t result_buffer_size) {
  // measure lookup & scan
  if (args.rep_lookup > 0 || args.rep_scan > 0) {
    timer_type lookup_timer, scan_timer;
    prefill(adapter, h_keys, h_key_lengths, h_values, args.keylen_max, args.num_prefill);
    #if !defined(NOGPU)
    auto d_lookup_keys = thrust::device_vector<key_slice_type>(h_lookup_keys.begin(), h_lookup_keys.end());
    auto d_lookup_key_lengths = thrust::device_vector<size_type>(h_lookup_key_lengths.begin(), h_lookup_key_lengths.end());
    auto d_results = thrust::device_vector<value_type>(result_buffer_size);
    auto d_scan_upper_keys_if_btree = thrust::device_vector<key_slice_type>(h_scan_upper_keys_if_btree.begin(), h_scan_upper_keys_if_btree.end());
    #else
    std::vector<value_type> h_results(result_buffer_size);
    #endif
    for (uint32_t r = 0; r < args.rep_lookup; r++) {
      lookup_timer.start();
      #if !defined(NOGPU)
      adapter.find(d_lookup_keys.data().get(), args.keylen_max, d_lookup_key_lengths.data().get(), d_results.data().get(), args.num_lookups);
      #else
      helper_multithread([&](std::size_t task_idx) {
          h_results[task_idx] = adapter.find(&h_lookup_keys[task_idx * args.keylen_max], h_lookup_key_lengths[task_idx]);
        }, args.num_lookups,
        [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
      #endif
      lookup_timer.stop();
      #if !defined(NOGPU)
      cuda_try(cudaDeviceSynchronize());
      #endif
      lookup_timer.record();
    }
    if constexpr (adapter_type::is_ordered) {
      for (uint32_t r = 0; r < args.rep_scan; r++) {
        scan_timer.start();
        #if !defined(NOGPU)
        adapter.scan(d_lookup_keys.data().get(), args.keylen_max, d_lookup_key_lengths.data().get(), args.scan_count, d_results.data().get(), args.num_scans, d_scan_upper_keys_if_btree.data().get());
        #else
        helper_multithread([&](std::size_t task_idx) {
            adapter.scan(&h_lookup_keys[task_idx * args.keylen_max], h_lookup_key_lengths[task_idx], args.scan_count, &h_results[task_idx * args.scan_count]);
          }, args.num_scans,
          [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
        #endif
        scan_timer.stop();
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        scan_timer.record();
      }
    }
    adapter.destroy();
    if (args.rep_lookup > 0) {
      lookup_timer.print_rate_Mops("lookup", args.num_lookups);
    }
    if (adapter_type::is_ordered && args.rep_scan > 0) {
      scan_timer.print_rate_Mops("scan", args.num_scans);
    }
  }

  // measure insert & delete
  if (args.rep_insdel > 0) {
    check_argument(args.num_prefill > args.num_insdel);
    timer_type insert_timer, delete_timer;
    for (uint32_t r = 0; r < args.rep_insdel; r++) {
      prefill(adapter, h_keys, h_key_lengths, h_values, args.keylen_max, args.num_prefill);
      {
        #if !defined(NOGPU)
        auto d_insert_keys = thrust::device_vector<key_slice_type>(
          h_keys.begin() + (args.keylen_max * args.num_prefill),
          h_keys.begin() + (args.keylen_max * (args.num_prefill + args.num_insdel)));
        auto d_insert_key_lengths = thrust::device_vector<size_type>(
          h_key_lengths.begin() + args.num_prefill,
          h_key_lengths.begin() + (args.num_prefill + args.num_insdel));
        auto d_insert_values = thrust::device_vector<value_type>(
          h_values.begin() + args.num_prefill,
          h_values.begin() + (args.num_prefill + args.num_insdel));
        #endif
        insert_timer.start();
        #if !defined(NOGPU)
        adapter.insert(d_insert_keys.data().get(), args.keylen_max, d_insert_key_lengths.data().get(), d_insert_values.data().get(), args.num_insdel);
        #else
        helper_multithread([&](std::size_t task_idx) {
            adapter.insert(&h_keys[(args.num_prefill + task_idx) * args.keylen_max], h_key_lengths[args.num_prefill + task_idx], h_values[args.num_prefill + task_idx]);
          }, args.num_insdel,
          [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
        #endif
        insert_timer.stop();
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        insert_timer.record();
      }
      {
        #if !defined(NOGPU)
        auto d_delete_keys = thrust::device_vector<key_slice_type>(
          h_keys.begin(), h_keys.begin() + (args.keylen_max * args.num_insdel));
        auto d_delete_key_lengths = thrust::device_vector<size_type>(
          h_key_lengths.begin(), h_key_lengths.begin() + args.num_insdel);
        #endif
        delete_timer.start();
        #if !defined(NOGPU)
        adapter.erase(d_delete_keys.data().get(), args.keylen_max, d_delete_key_lengths.data().get(), args.num_insdel);
        #else
        helper_multithread([&](std::size_t task_idx) {
            adapter.erase(&h_keys[task_idx * args.keylen_max], h_key_lengths[task_idx]);
          }, args.num_insdel,
          [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
        #endif
        delete_timer.stop();
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        delete_timer.record();
      }
      adapter.destroy();
    }
    insert_timer.print_rate_Mops("insert", args.num_insdel);
    delete_timer.print_rate_Mops("delete", args.num_insdel);
  }

  // measure mixed
  if constexpr (adapter_type::support_mixed) {
    if (args.rep_mixed > 0) {
      check_argument(args.num_prefill > args.num_mixed);
      timer_type mix_timer;
      for (uint32_t r = 0; r < args.rep_mixed; r++) {
        prefill(adapter, h_keys, h_key_lengths, h_values, args.keylen_max, args.num_prefill);
        #if !defined(NOGPU)
        auto d_mix_types = thrust::device_vector<kernels::request_type>(
          h_mix_types.begin(), h_mix_types.end());
        auto d_mix_keys = thrust::device_vector<key_slice_type>(
          h_mix_keys.begin(), h_mix_keys.end());
        auto d_mix_key_lengths = thrust::device_vector<size_type>(
          h_mix_key_lengths.begin(), h_mix_key_lengths.end());
        auto d_mix_values = thrust::device_vector<value_type>(
          h_mix_values.begin(), h_mix_values.end());
        #endif
        mix_timer.start();
        #if !defined(NOGPU)
        adapter.mixed_batch(d_mix_types.data().get(), d_mix_keys.data().get(), args.keylen_max, d_mix_key_lengths.data().get(), d_mix_values.data().get(), args.num_mixed);
        #else
        helper_multithread([&](std::size_t task_idx) {
            if (h_mix_types[task_idx] == kernels::request_type_find) {
              h_mix_values[task_idx] = adapter.find(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx]);
            }
            else if (h_mix_types[task_idx] == kernels::request_type_insert) {
              adapter.insert(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx], h_values[task_idx]);
            }
            else {
              adapter.erase(&h_mix_keys[task_idx * args.keylen_max], h_mix_key_lengths[task_idx]);
            }
          }, args.num_mixed,
          [&]() { adapter.thread_enter(); }, [&]() { adapter.thread_exit(); });
        #endif
        mix_timer.stop();
        #if !defined(NOGPU)
        cuda_try(cudaDeviceSynchronize());
        #endif
        mix_timer.record();
        adapter.destroy();
      }
      mix_timer.print_rate_Mops("mixed", args.num_mixed);
    }
  }
}

} // namespace universal_bench

int main(int argc, char** argv) {
  auto arg_strings = std::vector<std::string>(argv, argv + argc);
  universal::args_type args(arg_strings);

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
  check_argument(args.rep_lookup > 0 || args.rep_scan > 0 ||
                 args.rep_insdel > 0 || args.rep_mixed > 0);

  // generate keys and queries
  std::vector<key_slice_type> h_keys;
  std::vector<size_type> h_key_lengths;
  std::vector<value_type> h_values;
  std::vector<key_slice_type> h_lookup_keys;
  std::vector<size_type> h_lookup_key_lengths;
  std::vector<key_slice_type> h_scan_upper_keys_if_btree;
  std::vector<kernels::request_type> h_mix_types;
  std::vector<key_slice_type> h_mix_keys;
  std::vector<size_type> h_mix_key_lengths;
  std::vector<value_type> h_mix_values;
  // 1. generate keys: (num_prefill + max(num_insdel, num_mixed))
  std::size_t num_keys = args.num_prefill + std::max<std::size_t>(
    (args.rep_insdel > 0) ? args.num_insdel : 0,
    (args.rep_mixed > 0) ? args.num_mixed : 0);
  universal::generate_key_values(
    h_keys, h_key_lengths, h_values,
    num_keys, args.keylen_prefix, args.keylen_min, args.keylen_max, args.keylen_theta,
    false);
  // 2. generate lookup keys: max(num_lookups, num_scans)
  std::size_t num_lookups_keys = std::max<std::size_t>(
    (args.rep_lookup > 0) ? args.num_lookups : 0,
    (args.rep_scan > 0) ? args.num_scans : 0);
  if (num_lookups_keys > 0) {
    universal::generate_lookup_keys(
      h_lookup_keys, h_lookup_key_lengths, h_keys, h_key_lengths,
      args.num_prefill, args.keylen_max, num_lookups_keys, args.lookup_theta);
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
    (args.rep_lookup > 0) ? args.num_lookups : 0,
    (args.rep_scan > 0) ? (scan_result_buffer_multiplier * args.num_scans * args.scan_count) : 0);
  // 4. generate mixed keys
  if (args.rep_mixed > 0) {
    universal::generate_mixed_keys(
      h_mix_types, h_mix_keys, h_mix_key_lengths, h_mix_values, h_keys, h_key_lengths, h_values,
      args.num_prefill, args.keylen_max, args.num_mixed, args.mix_read_ratio, args.mix_presort, args.lookup_theta);
  }

  // run benchmark
  #define ADAPTER_RUN_BENCH(index) \
  if (args.index_type == #index) { \
    universal::run_bench(index##_adapter_, \
      h_keys, h_key_lengths, h_values, h_lookup_keys, h_lookup_key_lengths, h_scan_upper_keys_if_btree, \
      h_mix_types, h_mix_keys, h_mix_key_lengths, h_mix_values, args, result_buffer_size); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH
}
