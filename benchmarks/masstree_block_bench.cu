#include <gpu_masstree.hpp>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <rkg.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using key_slice_type = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;
using allocator_type = simple_slab_allocator<128>;
using reclaimer_type = simple_debra_reclaimer<1024>;
using masstree_type = GpuMasstree::gpu_masstree<allocator_type, reclaimer_type, 32>;
using timings = std::array<double, 6>;
constexpr std::array<const char*, 6> operation_names = {"insert", "find", "scan", "update", "erase", "mixed"};

struct options {
  size_type num_keys = 100000;
  int device = 0;
  size_type min_key_length = 1;
  size_type max_key_length = 1;
  size_type min_value_length = 1;
  size_type max_value_length = 1;
  size_type max_counts_per_query = 16;
  float common_prefix_ratio = 0.1f;
  float erase_ratio = 1.0f;
  float allocator_pool_ratio = 0.9f;
  float mixed_insert_ratio = 0.25f;
  float mixed_update_ratio = 0.25f;
  float mixed_erase_ratio = 0.25f;
  size_type num_experiments = 3;
  size_type num_warmups = 1;
  size_type seed = 0;
  bool validate_result = false;
  bool verbose = false;
};

std::size_t checked_elements(std::initializer_list<std::size_t> dimensions) {
  std::size_t count = 1;
  for (auto dimension : dimensions) {
    if (dimension > std::numeric_limits<size_type>::max() / count) {
      throw std::invalid_argument("workload exceeds the kernel's 32-bit indexing range");
    }
    count *= dimension;
  }
  return count;
}

options parse_options(int argc, char** argv) {
  const auto arguments = std::vector<std::string>(argv, argv + argc);
  options opts;
  opts.num_keys = get_arg_value<size_type>(arguments, "num-keys").value_or(opts.num_keys);
  opts.device = get_arg_value<int>(arguments, "device").value_or(opts.device);
  opts.min_key_length = get_arg_value<size_type>(arguments, "min-key-length").value_or(opts.min_key_length);
  opts.max_key_length = get_arg_value<size_type>(arguments, "max-key-length").value_or(opts.max_key_length);
  opts.min_value_length = get_arg_value<size_type>(arguments, "min-value-length").value_or(opts.min_value_length);
  opts.max_value_length = get_arg_value<size_type>(arguments, "max-value-length").value_or(opts.max_value_length);
  opts.max_counts_per_query = get_arg_value<size_type>(arguments, "max-counts-per-query").value_or(opts.max_counts_per_query);
  opts.common_prefix_ratio = get_arg_value<float>(arguments, "common-prefix-ratio").value_or(opts.common_prefix_ratio);
  opts.erase_ratio = get_arg_value<float>(arguments, "erase-ratio").value_or(opts.erase_ratio);
  opts.allocator_pool_ratio = get_arg_value<float>(arguments, "allocator-pool-ratio").value_or(opts.allocator_pool_ratio);
  opts.mixed_insert_ratio = get_arg_value<float>(arguments, "mixed-insert-ratio").value_or(opts.mixed_insert_ratio);
  opts.mixed_update_ratio = get_arg_value<float>(arguments, "mixed-update-ratio").value_or(opts.mixed_update_ratio);
  opts.mixed_erase_ratio = get_arg_value<float>(arguments, "mixed-erase-ratio").value_or(opts.mixed_erase_ratio);
  opts.num_experiments = get_arg_value<size_type>(arguments, "num-experiments").value_or(opts.num_experiments);
  opts.num_warmups = get_arg_value<size_type>(arguments, "num-warmups").value_or(opts.num_warmups);
  opts.seed = get_arg_value<size_type>(arguments, "seed").value_or(opts.seed);
  opts.validate_result = get_arg_value<bool>(arguments, "validate-result").value_or(opts.validate_result);
  opts.verbose = get_arg_value<bool>(arguments, "verbose").value_or(opts.verbose);
  auto valid_ratio = [](float ratio) { return std::isfinite(ratio) && ratio >= 0.f && ratio <= 1.f; };
  if (opts.num_keys == 0 || opts.num_keys > std::numeric_limits<size_type>::max() / 2 - reclaimer_type::block_size_ ||
      opts.num_experiments == 0 || opts.max_counts_per_query == 0 ||
      opts.min_key_length == 0 || opts.min_key_length > opts.max_key_length ||
      opts.min_value_length == 0 || opts.min_value_length > opts.max_value_length ||
      opts.max_key_length > 65535 || opts.max_value_length > 65535) {
    throw std::invalid_argument("invalid request count or key/value length range");
  }
  if (!valid_ratio(opts.common_prefix_ratio) || opts.common_prefix_ratio == 0.f ||
      !valid_ratio(opts.allocator_pool_ratio) || opts.allocator_pool_ratio == 0.f || opts.allocator_pool_ratio == 1.f ||
      !valid_ratio(opts.erase_ratio) || !valid_ratio(opts.mixed_insert_ratio) ||
      !valid_ratio(opts.mixed_update_ratio) || !valid_ratio(opts.mixed_erase_ratio) ||
      static_cast<double>(opts.mixed_insert_ratio) + opts.mixed_update_ratio + opts.mixed_erase_ratio > 1.000001) {
    throw std::invalid_argument("invalid ratio or mixed ratios exceed one");
  }
  checked_elements({2u, opts.num_keys, opts.max_key_length});
  checked_elements({2u, opts.num_keys, opts.max_value_length});
  checked_elements({opts.num_keys, opts.max_counts_per_query, opts.max_value_length});
  return opts;
}

size_type num_erases(const options& opts) {
  return static_cast<size_type>(static_cast<double>(opts.num_keys) * opts.erase_ratio);
}

struct workload {
  std::vector<key_slice_type> keys;
  std::vector<size_type> key_lengths;
  std::vector<value_type> values;
  std::vector<value_type> updated_values;
  std::vector<size_type> value_lengths;

  explicit workload(const options& opts) {
    std::mt19937 rng(opts.seed);
    const size_type total_keys = 2 * opts.num_keys;
    rkg::generate_varlen_keys<key_slice_type, size_type>(
      keys, key_lengths, total_keys, opts.min_key_length, opts.max_key_length, rng,
      rkg::distribution_type::unique_random, opts.common_prefix_ratio);
    values.resize(static_cast<std::size_t>(total_keys) * opts.max_value_length);
    updated_values.resize(values.size());
    value_lengths.resize(total_keys);
    std::uniform_int_distribution<size_type> lengths(opts.min_value_length, opts.max_value_length);
    for (size_type key_id = 0; key_id < total_keys; key_id++) {
      value_lengths[key_id] = lengths(rng);
      for (size_type slice = 0; slice < value_lengths[key_id]; slice++) {
        const auto offset = static_cast<std::size_t>(key_id) * opts.max_value_length + slice;
        values[offset] = (key_id + 1) ^ (slice * 0x9e3779b9u);
        updated_values[offset] = values[offset] ^ 0x80000000u;
      }
    }
  }
};

struct mixed_workload {
  std::vector<kernels::request_type> types;
  std::vector<key_slice_type> keys;
  std::vector<size_type> key_lengths;
  std::vector<value_type> values;
  std::vector<size_type> value_lengths;
  std::vector<size_type> key_ids;

  mixed_workload(const workload& input, const options& opts)
      : types(opts.num_keys), keys(static_cast<std::size_t>(opts.num_keys) * opts.max_key_length),
        key_lengths(opts.num_keys), values(static_cast<std::size_t>(opts.num_keys) * opts.max_value_length),
        value_lengths(opts.num_keys), key_ids(opts.num_keys) {
    const size_type inserts = static_cast<size_type>(static_cast<double>(opts.num_keys) * opts.mixed_insert_ratio);
    const size_type updates = static_cast<size_type>(static_cast<double>(opts.num_keys) * opts.mixed_update_ratio);
    const size_type erases = static_cast<size_type>(static_cast<double>(opts.num_keys) * opts.mixed_erase_ratio);
    if (static_cast<uint64_t>(inserts) + updates + erases > opts.num_keys) {
      throw std::invalid_argument("rounded mixed request counts exceed num-keys");
    }
    std::vector<size_type> order(opts.num_keys);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(opts.seed);
    std::shuffle(order.begin(), order.end(), rng);
    for (size_type source = 0; source < opts.num_keys; source++) {
      const size_type destination = order[source];
      const bool is_insert = source < inserts;
      const size_type key_id = is_insert ? opts.num_keys + source : source - inserts;
      types[destination] = is_insert ? kernels::request_type_insert :
          source < inserts + updates ? kernels::request_type_update :
          source < inserts + updates + erases ? kernels::request_type_erase : kernels::request_type_find;
      key_ids[destination] = key_id;
      key_lengths[destination] = input.key_lengths[key_id];
      std::copy_n(input.keys.data() + static_cast<std::size_t>(key_id) * opts.max_key_length,
                  opts.max_key_length, keys.data() + static_cast<std::size_t>(destination) * opts.max_key_length);
      if (types[destination] == kernels::request_type_insert || types[destination] == kernels::request_type_update) {
        value_lengths[destination] = input.value_lengths[key_id];
        const auto& source_values = is_insert ? input.values : input.updated_values;
        std::copy_n(source_values.data() + static_cast<std::size_t>(key_id) * opts.max_value_length,
                    opts.max_value_length, values.data() + static_cast<std::size_t>(destination) * opts.max_value_length);
      }
    }
  }
};

struct device_workload {
  thrust::device_vector<key_slice_type> keys;
  thrust::device_vector<size_type> key_lengths;
  thrust::device_vector<value_type> values, updated_values, query_values, scan_values;
  thrust::device_vector<size_type> value_lengths, query_lengths, scan_lengths, scan_counts;

  device_workload(const workload& input, const options& opts)
      : keys(input.keys.begin(), input.keys.end()), key_lengths(input.key_lengths.begin(), input.key_lengths.end()),
        values(input.values.begin(), input.values.end()), updated_values(input.updated_values.begin(), input.updated_values.end()),
        query_values(input.values.size()),
        scan_values(checked_elements({opts.num_keys, opts.max_counts_per_query, opts.max_value_length})),
        value_lengths(input.value_lengths.begin(), input.value_lengths.end()), query_lengths(input.value_lengths.size()),
        scan_lengths(checked_elements({opts.num_keys, opts.max_counts_per_query})), scan_counts(opts.num_keys) {}
};

struct device_mixed_workload {
  thrust::device_vector<kernels::request_type> types;
  thrust::device_vector<key_slice_type> keys;
  thrust::device_vector<size_type> key_lengths;
  thrust::device_vector<value_type> values;
  thrust::device_vector<size_type> value_lengths;
  thrust::device_vector<bool> results;

  explicit device_mixed_workload(const mixed_workload& input)
      : types(input.types.begin(), input.types.end()), keys(input.keys.begin(), input.keys.end()),
        key_lengths(input.key_lengths.begin(), input.key_lengths.end()), values(input.values.begin(), input.values.end()),
        value_lengths(input.value_lengths.begin(), input.value_lengths.end()), results(input.types.size(), false) {}
};

template <typename device_func>
void launch_single_cta(masstree_type& tree, const device_func& func, size_type count) {
  if (count == 0) { return; }
  constexpr bool reclaim = device_func::reclaim_required;
  constexpr std::size_t shmem_bytes = reclaim ? sizeof(size_type) * masstree_type::device_reclaimer_context_type::required_shmem_size() : 0;
  kernels::batch_kernel<reclaim, 32, false><<<1, reclaimer_type::block_size_, shmem_bytes>>>(tree, func, count);
  cuda_try(cudaGetLastError());
}

template <typename device_func>
double time_single_cta(masstree_type& tree, const device_func& func, size_type count) {
  if (count == 0) { return 0.; }
  gpu_timer timer;
  timer.start_timer();
  launch_single_cta(tree, func, count);
  timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());
  return timer.get_elapsed_ms();
}

void check_value(const value_type* actual, size_type actual_length,
                 const value_type* expected, size_type expected_length, const std::string& label) {
  if (actual_length != expected_length) {
    throw std::runtime_error(label + ": value length mismatch");
  }
  if (!std::equal(expected, expected + expected_length, actual)) {
    throw std::runtime_error(label + ": value mismatch");
  }
}

void validate_find_results(const device_workload& data, const std::vector<value_type>& expected_values,
                           const std::vector<size_type>& expected_lengths, size_type count,
                           const options& opts, const std::string& label) {
  const thrust::host_vector<value_type> values(data.query_values);
  const thrust::host_vector<size_type> lengths(data.query_lengths);
  for (size_type query = 0; query < count; query++) {
    const auto offset = static_cast<std::size_t>(query) * opts.max_value_length;
    check_value(values.data() + offset, lengths[query], expected_values.data() + offset,
                 expected_lengths[query], label + " key " + std::to_string(query));
  }
}

void validate_scan_results(const device_workload& data, const workload& input, const options& opts) {
  std::vector<size_type> order(opts.num_keys);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](size_type left, size_type right) {
    const auto left_begin = input.keys.begin() + static_cast<std::size_t>(left) * opts.max_key_length;
    const auto right_begin = input.keys.begin() + static_cast<std::size_t>(right) * opts.max_key_length;
    return std::lexicographical_compare(left_begin, left_begin + input.key_lengths[left],
                                        right_begin, right_begin + input.key_lengths[right]);
  });
  std::vector<size_type> ranks(opts.num_keys);
  for (size_type rank = 0; rank < opts.num_keys; rank++) { ranks[order[rank]] = rank; }
  const thrust::host_vector<value_type> values(data.scan_values);
  const thrust::host_vector<size_type> lengths(data.scan_lengths);
  const thrust::host_vector<size_type> counts(data.scan_counts);
  for (size_type query = 0; query < opts.num_keys; query++) {
    const size_type count = std::min(opts.max_counts_per_query, opts.num_keys - ranks[query]);
    if (counts[query] != count) {
      throw std::runtime_error("scan query " + std::to_string(query) + ": count mismatch");
    }
    for (size_type result = 0; result < count; result++) {
      const auto slot = static_cast<std::size_t>(query) * opts.max_counts_per_query + result;
      const auto key_id = order[ranks[query] + result];
      check_value(values.data() + slot * opts.max_value_length, lengths[slot],
                   input.values.data() + static_cast<std::size_t>(key_id) * opts.max_value_length,
                   input.value_lengths[key_id], "scan query " + std::to_string(query) + " result " + std::to_string(result));
    }
  }
}

void validate_mixed_results(const device_mixed_workload& requests, const mixed_workload& mixed,
                            const workload& input, const options& opts) {
  const thrust::host_vector<value_type> values(requests.values);
  const thrust::host_vector<size_type> lengths(requests.value_lengths);
  const thrust::host_vector<bool> results(requests.results);
  for (size_type request = 0; request < opts.num_keys; request++) {
    if (mixed.types[request] == kernels::request_type_find) {
      const auto key_id = mixed.key_ids[request];
      check_value(values.data() + static_cast<std::size_t>(request) * opts.max_value_length, lengths[request],
                   input.values.data() + static_cast<std::size_t>(key_id) * opts.max_value_length,
                   input.value_lengths[key_id], "mixed find " + std::to_string(request));
    }
    else if (!results[request]) {
      throw std::runtime_error("mixed mutation " + std::to_string(request) + " failed");
    }
  }
}

template <cuda::thread_scope scope>
timings run_trial(const workload& input, device_workload& data, const mixed_workload& mixed, const options& opts) {
  using insert_func = kernels::GpuMasstree::insert_device_func<masstree_type, false, true, scope>;
  using find_func = kernels::GpuMasstree::find_device_func<masstree_type, true, scope>;
  using scan_func = kernels::GpuMasstree::scan_device_func<masstree_type, false, true, scope>;
  using update_func = kernels::GpuMasstree::update_device_func<masstree_type, true, scope>;
  using erase_func = kernels::GpuMasstree::erase_device_func<masstree_type, true, true, true, true, scope>;
  using mixed_func = kernels::GpuMasstree::mixed_device_func<masstree_type, true, true, true, true, scope>;
  const insert_func insert{data.keys.data().get(), opts.max_key_length, data.key_lengths.data().get(),
                           data.values.data().get(), opts.max_value_length, data.value_lengths.data().get()};
  timings elapsed{};
  {
    allocator_type allocator(opts.allocator_pool_ratio);
    reclaimer_type reclaimer;
    masstree_type tree(allocator, reclaimer);
    const find_func find{data.keys.data().get(), opts.max_key_length, data.key_lengths.data().get(),
                         data.query_values.data().get(), opts.max_value_length, data.query_lengths.data().get()};
    const scan_func scan{data.keys.data().get(), data.key_lengths.data().get(), opts.max_key_length,
                         opts.max_counts_per_query, nullptr, nullptr, data.scan_counts.data().get(),
                         data.scan_values.data().get(), data.scan_lengths.data().get(), opts.max_value_length, nullptr, nullptr};
    const update_func update{data.keys.data().get(), opts.max_key_length, data.key_lengths.data().get(),
                             data.updated_values.data().get(), opts.max_value_length, data.value_lengths.data().get()};
    const erase_func erase{data.keys.data().get(), opts.max_key_length, data.key_lengths.data().get()};
    elapsed[0] = time_single_cta(tree, insert, opts.num_keys);
    elapsed[1] = time_single_cta(tree, find, opts.num_keys);
    if (opts.validate_result) {
      validate_find_results(data, input.values, input.value_lengths, opts.num_keys, opts, "insert/find");
    }
    elapsed[2] = time_single_cta(tree, scan, opts.num_keys);
    if (opts.validate_result) { validate_scan_results(data, input, opts); }
    elapsed[3] = time_single_cta(tree, update, opts.num_keys);
    if (opts.validate_result) {
      launch_single_cta(tree, find, opts.num_keys);
      cuda_try(cudaDeviceSynchronize());
      validate_find_results(data, input.updated_values, input.value_lengths, opts.num_keys, opts, "update");
    }
    elapsed[4] = time_single_cta(tree, erase, num_erases(opts));
    if (opts.validate_result) {
      auto expected_lengths = input.value_lengths;
      std::fill_n(expected_lengths.begin(), num_erases(opts), 0u);
      launch_single_cta(tree, find, opts.num_keys);
      cuda_try(cudaDeviceSynchronize());
      validate_find_results(data, input.updated_values, expected_lengths, opts.num_keys, opts, "erase/survivor");
    }
  }
  {
    device_mixed_workload requests(mixed);
    allocator_type allocator(opts.allocator_pool_ratio);
    reclaimer_type reclaimer;
    masstree_type tree(allocator, reclaimer);
    launch_single_cta(tree, insert, opts.num_keys);
    cuda_try(cudaDeviceSynchronize());
    const mixed_func func{requests.types.data().get(), requests.keys.data().get(), opts.max_key_length,
                          requests.key_lengths.data().get(), requests.values.data().get(), opts.max_value_length,
                          requests.value_lengths.data().get(), requests.results.data().get()};
    elapsed[5] = time_single_cta(tree, func, opts.num_keys);
    if (opts.validate_result) {
      validate_mixed_results(requests, mixed, input, opts);
      auto expected_values = input.values;
      std::vector<size_type> expected_lengths(input.value_lengths.size(), 0);
      std::copy_n(input.value_lengths.begin(), opts.num_keys, expected_lengths.begin());
      for (size_type request = 0; request < opts.num_keys; request++) {
        const auto key_id = mixed.key_ids[request];
        if (mixed.types[request] == kernels::request_type_insert) {
          expected_lengths[key_id] = input.value_lengths[key_id];
        }
        else if (mixed.types[request] == kernels::request_type_erase) {
          expected_lengths[key_id] = 0;
        }
        else if (mixed.types[request] == kernels::request_type_update) {
          const auto offset = static_cast<std::size_t>(key_id) * opts.max_value_length;
          std::copy_n(input.updated_values.data() + offset, opts.max_value_length, expected_values.data() + offset);
        }
      }
      const find_func find{data.keys.data().get(), opts.max_key_length, data.key_lengths.data().get(),
                           data.query_values.data().get(), opts.max_value_length, data.query_lengths.data().get()};
      launch_single_cta(tree, find, 2 * opts.num_keys);
      cuda_try(cudaDeviceSynchronize());
      validate_find_results(data, expected_values, expected_lengths, 2 * opts.num_keys, opts, "mixed final state");
    }
  }
  return elapsed;
}

void print_results(const std::array<timings, 2>& totals, const options& opts) {
  std::cout << std::left << std::setw(12) << "operation" << std::setw(12) << "requests"
            << std::setw(14) << "device_ms" << std::setw(14) << "block_ms"
            << std::setw(16) << "device_Mop/s" << std::setw(16) << "block_Mop/s" << "block/device\n";
  for (std::size_t operation = 0; operation < operation_names.size(); operation++) {
    const auto count = operation == 4 ? num_erases(opts) : opts.num_keys;
    std::cout << std::setw(12) << operation_names[operation] << std::setw(12) << count;
    if (count == 0) {
      std::cout << "skipped\n";
      continue;
    }
    const double device_ms = totals[0][operation] / opts.num_experiments;
    const double block_ms = totals[1][operation] / opts.num_experiments;
    std::cout << std::fixed << std::setprecision(4)
              << std::setw(14) << device_ms << std::setw(14) << block_ms
              << std::setw(16) << count / (device_ms * 1000.) << std::setw(16) << count / (block_ms * 1000.)
              << device_ms / block_ms << '\n';
  }
}

}

int main(int argc, char** argv) {
  try {
    const auto opts = parse_options(argc, argv);
    int device_count = 0;
    const auto device_result = cudaGetDeviceCount(&device_count);
    if (device_result != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA unavailable: ") + cudaGetErrorString(device_result));
    }
    if (opts.device < 0 || opts.device >= device_count) {
      throw std::invalid_argument("invalid CUDA device");
    }
    cuda_try(cudaSetDevice(opts.device));
    cudaDeviceProp properties;
    cuda_try(cudaGetDeviceProperties(&properties, opts.device));
    std::cout << "Device[" << opts.device << "]: " << properties.name << '\n'
              << "grid=1 block=" << reclaimer_type::block_size_ << " tile=32 num-keys=" << opts.num_keys
              << " seed=" << opts.seed << " scan-cap=" << opts.max_counts_per_query << '\n'
              << "mixed insert:update:erase:find=" << opts.mixed_insert_ratio << ':' << opts.mixed_update_ratio
              << ':' << opts.mixed_erase_ratio << ':'
              << 1.f - opts.mixed_insert_ratio - opts.mixed_update_ratio - opts.mixed_erase_ratio << '\n'
              << "Find and scan rates count queries; mutation timings include existing reclamation.\n";
    const workload input(opts);
    const mixed_workload mixed(input, opts);
    device_workload data(input, opts);
    std::array<timings, 2> totals{};
    auto run_scope = [&](bool block_scope, bool warmup, size_type trial) {
      const auto elapsed = block_scope ? run_trial<cuda::thread_scope_block>(input, data, mixed, opts) :
                                        run_trial<cuda::thread_scope_device>(input, data, mixed, opts);
      for (std::size_t operation = 0; operation < elapsed.size(); operation++) {
        if (!warmup) { totals[block_scope ? 1 : 0][operation] += elapsed[operation]; }
        if (opts.verbose) {
          std::cout << (warmup ? "warmup " : "trial ") << trial << ' ' << (block_scope ? "block " : "device ")
                    << operation_names[operation] << ' ' << elapsed[operation] << " ms\n";
        }
      }
    };
    for (size_type trial = 0; trial < opts.num_warmups; trial++) {
      run_scope(trial % 2 != 0, true, trial);
      run_scope(trial % 2 == 0, true, trial);
    }
    for (size_type trial = 0; trial < opts.num_experiments; trial++) {
      run_scope(trial % 2 != 0, false, trial);
      run_scope(trial % 2 == 0, false, trial);
    }
    print_results(totals, opts);
    if (opts.validate_result) { std::cout << "All results valid for both scopes.\n"; }
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
