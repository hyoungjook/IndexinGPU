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
#include <fstream>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <macros.hpp>

using key_slice_type = uint32_t;
using value_slice_type = uint32_t;
using size_type = uint32_t;

namespace universal {

template <class F, class ThreadEnter, class ThreadExit>
void helper_multithread(F&& f, std::size_t num_tasks, ThreadEnter&& thread_enter, ThreadExit&& thread_exit) {
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    workers.emplace_back([&](unsigned thread_id) {
      std::forward<ThreadEnter>(thread_enter)(thread_id);
      for (std::size_t task_idx = thread_id; task_idx < num_tasks; task_idx += num_workers) {
        std::forward<F>(f)(task_idx, thread_id);
      }
      std::forward<ThreadExit>(thread_exit)(thread_id);
    }, tid);
  }
  for (auto& w: workers) { w.join(); }
}

inline
std::size_t feistel_permute(std::size_t x, std::size_t N, uint64_t seed) {
  if (N <= 1) return 0;
  auto mix = [](std::uint64_t v) {
      v ^= v >> 33;
      v *= 0xff51afd7ed558ccdULL;
      v ^= v >> 33;
      v *= 0xc4ceb9fe1a85ec53ULL;
      v ^= v >> 33;
      return v;
  };
  unsigned half = 0;
  while ((std::size_t(1) << (2 * half)) < N) ++half;
  uint64_t mask = (static_cast<uint64_t>(1) << half) - 1;
  while (true) {
    uint64_t L = x >> half, R = x & mask;
    for (int r = 0; r < 4; ++r) {
      uint64_t t = L;
      L = R;
      R = t ^ (mix(R + seed + r) & mask);
    }
    x = (L << half) | R;
    if (x < N) {
      return x;
    }
  }
}

template <typename T>
struct zipfian_int_distribution {
  // theta = 0 equals to uniform distribution
 public:
  zipfian_int_distribution(T min_value, T max_value, double theta) 
      : min_value_(min_value)
      , max_value_(max_value)
      , theta_(theta) {
    check_argument(0.0 <= theta && theta < 1.0);
    check_argument(min_value <= max_value);
    uint64_t num_values = max_value_ - min_value_ + 1;
    zetan_ = zeta(num_values);
    alpha_ = 1.0 / (1.0 - theta_);
    eta_ = (1 - std::pow(2.0 / num_values, 1 - theta_)) / (1 - zeta(2) / zetan_);
  }
  T operator()(std::mt19937& rng) {
    std::uniform_real_distribution<double> double_dist(0.0, std::nextafter(1.0, 2.0));
    double u = double_dist(rng);
    double uz = u * zetan_;
    uint64_t base;
    if (uz < 1.0) { base = 0; }
    else if (uz < 1.0 + std::pow(0.5, theta_)) { base = 1; }
    else {
      uint64_t max_base = max_value_ - min_value_;
      base = static_cast<uint64_t>(max_base * std::pow(eta_ * u - eta_ + 1, alpha_));
    }
    return min_value_ + static_cast<T>(base);
  }
 private:
  double zeta(uint64_t n) {
    double sum = 0;
    for (uint64_t i = 0; i < n; i++) {
      sum += (1.0 / std::pow(i + 1, theta_));
    }
    return sum;
  }
  uint64_t min_value_, max_value_;
  double theta_, alpha_, eta_, zetan_;
};

inline
std::size_t key_hasher(const key_slice_type* key, size_type length) {
  std::size_t hash = 0;
  for (size_type i = 0; i < length; i++) {
    hash ^= std::hash<uint32_t>{}(key[i]) + 0x9e3779b9 + (hash<<6) + (hash>>2);
  }
  return hash;
}

inline
void load_keys_from_dataset(std::vector<key_slice_type>& keys,
                            std::vector<size_type>& key_lengths,
                            std::size_t& num_keys,
                            uint32_t& keylen_min,
                            uint32_t& keylen_max,
                            const std::string& dataset_file,
                            bool big_endian,
                            bool append_zero_at_end) {
  std::ifstream ifs(dataset_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed opening " << dataset_file << "..." << std::endl;
    std::abort();
  }
  std::vector<std::string> string_keys;
  std::string line;
  keylen_min = std::numeric_limits<uint32_t>::max();
  keylen_max = 0;
  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    auto key_length = static_cast<uint32_t>((line.size() + sizeof(key_slice_type) - 1) / sizeof(key_slice_type));
    string_keys.push_back(line);
    keylen_min = std::min(keylen_min, key_length);
    keylen_max = std::max(keylen_max, key_length);
  }
  num_keys = string_keys.size();
  check_argument(num_keys > 0);
  keys = std::vector<key_slice_type>(num_keys * keylen_max, 0);
  key_lengths = std::vector<size_type>(num_keys, 0);
  helper_multithread([&](std::size_t key_idx, [[maybe_unused]] unsigned thread_id) {
      const auto& string_key = string_keys[key_idx];
      auto* key = &keys[key_idx * keylen_max];
      uint32_t length = static_cast<uint32_t>((string_key.size() + sizeof(key_slice_type) - 1) / sizeof(key_slice_type));
      key_lengths[key_idx] = length;
      for (uint32_t slice = 0; slice < length; slice++) {
        key_slice_type key_slice = 0;
        for (uint32_t byte = 0; byte < sizeof(key_slice_type); byte++) {
          key_slice <<= 8;
          auto byte_idx = static_cast<std::size_t>(slice) * sizeof(key_slice_type) + byte;
          if (byte_idx < string_key.size()) {
            key_slice |= static_cast<key_slice_type>(static_cast<uint8_t>(string_key[byte_idx]));
          }
        }
        if (big_endian) {
          key_slice = __builtin_bswap32(key_slice);
        }
        key[slice] = key_slice;
      }
      if (append_zero_at_end && length < keylen_max && string_key.size() == length * sizeof(key_slice_type)) {
        // keylen_min might be wrong, but whatever
        key_lengths[key_idx]++;
        key[length] = 0;
      }
    }, num_keys, [](unsigned){}, [](unsigned){});
}

inline
void generate_values_from_keys(std::vector<value_slice_type>& values,
                               std::vector<size_type>& value_lengths,
                               const std::vector<key_slice_type>& keys,
                               const std::vector<size_type>& key_lengths,
                               std::size_t num_keys,
                               uint32_t keylen_max,
                               uint32_t valuelen_min,
                               uint32_t valuelen_max,
                               double valuelen_theta) {
  values = std::vector<value_slice_type>(num_keys * valuelen_max);
  value_lengths = std::vector<size_type>(num_keys);
  zipfian_int_distribution<key_slice_type> value_length_dist(valuelen_min, valuelen_max, valuelen_theta);
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::mt19937> per_thd_rng;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    per_thd_rng.emplace_back(tid + 1);
  }
  helper_multithread([&](std::size_t key_idx, unsigned thread_id) {
      auto& rng = per_thd_rng[thread_id];
      uint32_t value_length = value_length_dist(rng);
      value_lengths[key_idx] = value_length;
      value_slice_type value_slice = key_hasher(&keys[key_idx * keylen_max], key_lengths[key_idx]);
      auto* value = &values[key_idx * valuelen_max];
      for (uint32_t slice = 0; slice < value_length; slice++) {
        value[slice] = value_slice;
      }
    }, num_keys, [](unsigned){}, [](unsigned){});
}

inline
void generate_key_values(std::vector<key_slice_type>& keys,
                         std::vector<size_type>& key_lengths,
                         std::vector<value_slice_type>& values,
                         std::vector<size_type>& value_lengths,
                         std::size_t num_keys,
                         uint32_t keylen_prefix,
                         uint32_t keylen_min,
                         uint32_t keylen_max,
                         double keylen_theta,
                         uint32_t valuelen_min,
                         uint32_t valuelen_max,
                         double valuelen_theta,
                         bool big_endian) {
  // key: [common prefix (keylen_prefix)], [random slices], [unique_id]
  // normally unique_id is 1 slice, but if num_keys exceeds uint32, should be 2 slices
  // also spare uint32_max to represent non-existing key
  const uint32_t unique_slices = (num_keys < std::numeric_limits<key_slice_type>::max()) ? 1 : 2;
  check_argument(
    keylen_prefix + unique_slices <= keylen_min &&
    keylen_min <= keylen_max
  );
  // generate prefix
  std::mt19937 main_thd_rng(0);
  std::vector<key_slice_type> prefix(keylen_prefix);
  for (uint32_t slice = 0; slice < keylen_prefix; slice++) {
    std::uniform_int_distribution<key_slice_type> prefix_dist(0, std::numeric_limits<key_slice_type>::max());
    prefix[slice] = prefix_dist(main_thd_rng);
  }
  // generate keys
  keys = std::vector<key_slice_type>(num_keys * keylen_max);
  key_lengths = std::vector<size_type>(num_keys);
  values = std::vector<value_slice_type>(num_keys * valuelen_max);
  value_lengths = std::vector<size_type>(num_keys);
  zipfian_int_distribution<key_slice_type> length_dist(keylen_min, keylen_max, keylen_theta);
  zipfian_int_distribution<key_slice_type> value_length_dist(valuelen_min, valuelen_max, valuelen_theta);
  std::uniform_int_distribution<key_slice_type> slice_dist(0, std::numeric_limits<key_slice_type>::max());
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::mt19937> per_thd_rng;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    per_thd_rng.emplace_back(tid + 1);
  }
  helper_multithread([&](std::size_t key_idx, unsigned thread_id) {
      auto& rng = per_thd_rng[thread_id];
      auto* key = &keys[key_idx * keylen_max];
      // decide key length
      uint32_t length = length_dist(rng);
      key_lengths[key_idx] = length;
      // unique_id is random permutation of [0, num_keys)
      // this replaces single-threaded std::shuffle().
      auto unique_id = feistel_permute(key_idx, num_keys, 0);
      // fill slices
      for (uint32_t slice = 0; slice < length; slice++) {
        //  key[0:keylen_prefix) = prefix[]
        //  key[keylen_prefix:length-unique_slices) = random
        //  key[length-unique_slices:length-1] = key_idx
        if (slice < keylen_prefix) {
          key[slice] = prefix[slice];
        }
        else if (slice < length - unique_slices) {
          key[slice] = slice_dist(rng);
        }
        else if (slice == length - 2) {
          key[slice] = static_cast<key_slice_type>(unique_id >> (sizeof(key_slice_type) * 8));
        }
        else {
          key[slice] = static_cast<key_slice_type>(unique_id);
        }
      }
      // big endian
      if (big_endian) {
        for (uint32_t slice = 0; slice < length; slice++) {
          uint32_t key_slice = key[slice];
          key[slice] = __builtin_bswap32(key_slice);
        }
      }
      // compute value
      uint32_t value_length = value_length_dist(rng);
      value_slice_type value_slice = key_hasher(key, length);
      value_lengths[key_idx] = value_length;
      auto* value = &values[key_idx * valuelen_max];
      for (uint32_t slice = 0; slice < value_length; slice++) {
        value[slice] = value_slice;
      }
  }, num_keys, [](unsigned){}, [](unsigned){});
}

inline
void generate_lookup_keys(std::vector<key_slice_type>& lookup_keys,
                          std::vector<size_type>& lookup_key_lengths,
                          std::vector<key_slice_type>& keys,
                          std::vector<size_type>& key_lengths,
                          std::size_t num_keys,
                          uint32_t keylen_max,
                          std::size_t num_queries,
                          double lookup_theta) {
  // randomly select lookup key from given keys
  lookup_keys = std::vector<key_slice_type>(num_queries * keylen_max);
  lookup_key_lengths = std::vector<size_type>(num_queries);
  zipfian_int_distribution<std::size_t> key_choose_dist(0, num_keys - 1, lookup_theta);
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::mt19937> per_thd_rng;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    per_thd_rng.emplace_back(tid + 1);
  }
  helper_multithread([&](std::size_t lookup_idx, unsigned thread_id) {
    auto& rng = per_thd_rng[thread_id];
    auto* lookup_key = &lookup_keys[lookup_idx * keylen_max];
    // copy randomly from key
    std::size_t key_idx = key_choose_dist(rng);
    uint32_t length = key_lengths[key_idx];
    lookup_key_lengths[lookup_idx] = length;
    memcpy(lookup_key, &keys[key_idx * keylen_max], sizeof(key_slice_type) * length);
  }, num_queries, [](unsigned){}, [](unsigned){});
}

inline std::size_t mix_get_num_insdel(std::size_t num_mixed, double mix_read_ratio) {
  std::size_t num_lookups_tmp = static_cast<std::size_t>(mix_read_ratio * num_mixed);
  std::size_t num_insdel = (num_mixed - num_lookups_tmp) / 2;
  return num_insdel;
}

inline std::size_t mix_get_num_lookups(std::size_t num_mixed, double mix_read_ratio) {
  std::size_t num_insdel = mix_get_num_insdel(num_mixed, mix_read_ratio);
  std::size_t num_lookups = num_mixed - num_insdel * 2;
  return num_lookups;
}

inline
void generate_mixed_keys(std::vector<kernels::request_type>& mix_types,
                         std::vector<key_slice_type>& mix_keys,
                         std::vector<size_type>& mix_key_lengths,
                         std::vector<value_slice_type>& mix_values,
                         std::vector<size_type>& mix_value_lengths,
                         std::vector<std::size_t>& mix_key_tuple_ids,
                         std::vector<key_slice_type>& keys,
                         std::vector<size_type>& key_lengths,
                         std::vector<value_slice_type>& values,
                         std::vector<size_type>& value_lengths,
                         std::size_t num_keys,
                         uint32_t keylen_max,
                         uint32_t valuelen_max,
                         std::size_t num_mixed,
                         double mix_read_ratio,
                         bool mix_presort,
                         double lookup_theta) {
  check_argument(0 <= mix_read_ratio && mix_read_ratio <= 1.0);
  auto num_lookups = mix_get_num_lookups(num_mixed, mix_read_ratio);
  auto num_insdel = mix_get_num_insdel(num_mixed, mix_read_ratio);
  std::vector<std::size_t> shuffle_order(num_mixed);
  for (std::size_t i = 0; i < num_mixed; i++) shuffle_order[i] = i;
  if (!mix_presort) {
    std::mt19937 rng(0);
    std::shuffle(shuffle_order.begin(), shuffle_order.end(), rng);
  }
  mix_types = std::vector<kernels::request_type>(num_mixed);
  mix_keys = std::vector<key_slice_type>(num_mixed * keylen_max);
  mix_key_lengths = std::vector<size_type>(num_mixed);
  mix_values = std::vector<value_slice_type>(num_mixed * valuelen_max);
  mix_value_lengths = std::vector<size_type>(num_mixed);
  mix_key_tuple_ids = std::vector<std::size_t>(num_mixed);
  std::vector<key_slice_type> tmp_lookup_keys;
  std::vector<size_type> tmp_lookup_key_lengths;
  if (num_lookups > 0) {
    generate_lookup_keys(tmp_lookup_keys, tmp_lookup_key_lengths, keys, key_lengths,
      num_keys, keylen_max, num_lookups, lookup_theta);
  }
  helper_multithread([&](std::size_t idx, [[maybe_unused]] unsigned thread_id) {
    auto dst_idx = shuffle_order[idx];
    if (idx < num_lookups) {
      mix_types[dst_idx] = kernels::request_type_find;
      mix_key_lengths[dst_idx] = tmp_lookup_key_lengths[idx];
      memcpy(&mix_keys[dst_idx * keylen_max], &tmp_lookup_keys[idx * keylen_max], sizeof(key_slice_type) * keylen_max);
    }
    else if (idx < num_lookups + num_insdel) {
      mix_types[dst_idx] = kernels::request_type_insert;
      auto insert_idx = num_keys - 1 - (idx - num_lookups);
      mix_key_lengths[dst_idx] = key_lengths[insert_idx];
      memcpy(&mix_keys[dst_idx * keylen_max], &keys[insert_idx * keylen_max], sizeof(key_slice_type) * keylen_max);
      mix_value_lengths[dst_idx] = value_lengths[insert_idx];
      memcpy(&mix_values[dst_idx * valuelen_max], &values[insert_idx * valuelen_max], sizeof(value_slice_type) * valuelen_max);
      mix_key_tuple_ids[dst_idx] = insert_idx;
    }
    else {
      mix_types[dst_idx] = kernels::request_type_erase;
      auto delete_idx = idx - (num_lookups + num_insdel);
      mix_key_lengths[dst_idx] = key_lengths[delete_idx];
      memcpy(&mix_keys[dst_idx * keylen_max], &keys[delete_idx * keylen_max], sizeof(key_slice_type) * keylen_max);
    }
  }, num_mixed, [](unsigned){}, [](unsigned){});
}

#define FORALL_ARGUMENTS(x) \
  /* key distribution */ \
  x(max_keys, std::size_t, 10000000) \
  x(keylen_prefix, uint32_t, 0) \
  x(keylen_min, uint32_t, 1) \
  x(keylen_max, uint32_t, 1) \
  x(keylen_theta, double, 0.0) \
  x(dataset_file, std::string, "") \
  /* value distribution */ \
  x(valuelen_min, uint32_t, 1) \
  x(valuelen_max, uint32_t, 1) \
  x(valuelen_theta, double, 0.0) \
  /* lookup test */ \
  x(num_lookups, uint32_t, 0) \
  x(lookup_theta, double, 0.0) \
  /* scan test */ \
  x(num_scans, uint32_t, 0) \
  x(scan_count, uint32_t, 1) \
  /* insert delete test */ \
  x(num_insdel, uint32_t, 0) \
  /* mixed test */ \
  x(num_mixed, uint32_t, 0) \
  x(mix_read_ratio, double, 0.5) \
  x(mix_presort, bool, true) \
  /* space test */ \
  x(num_space, uint32_t, 0) \
  /* repeats */ \
  x(rep_lookup, uint32_t, 0) \
  x(rep_scan, uint32_t, 0) \
  x(rep_insdel, uint32_t, 0) \
  x(rep_mixed, uint32_t, 0) \
  x(rep_space, uint32_t, 0) \
  /* index config */ \
  x(index_type, std::string, "gpu_masstree") \
  /* etc */ \
  x(only_check_space, bool, false) \
  x(use_pinned_host_memory, bool, false)

struct args_type {
  #define DECLARE_ARGUMENT(arg, type, default_value) \
  type arg;
  FORALL_ARGUMENTS(DECLARE_ARGUMENT)
  #undef DECLARE_ARGUMENT

  args_type(std::vector<std::string>& arg_strings) {
    #define PARSE_ARGUMENT(arg, type, default_value) \
    arg = get_arg_value<type>(arg_strings, #arg).value_or(default_value);
    FORALL_ARGUMENTS(PARSE_ARGUMENT)
    #undef PARSE_ARGUMENT
  }

  void print() {
    std::cout << "arguments:" << std::endl;
    #define PRINT_ARGUMENT(arg, type, default_value) \
    std::cout << "  " #arg "=" << arg << std::endl;
    FORALL_ARGUMENTS(PRINT_ARGUMENT)
    #undef PRINT_ARGUMENT
  }
};

} // namespace universal
