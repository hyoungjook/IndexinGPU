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
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <ROWEX/Tree.h>

struct cpu_art_adapter {
  static constexpr bool is_ordered = true;
  static constexpr bool support_mixed = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void register_dataset(const key_slice_type* keys, const size_type* key_lengths, const value_type* values) {
    keys_ = keys;
    key_lengths_ = key_lengths;
    values_ = values;
    key_stride_ = configs_.keylen_max;
  }
  void initialize() {
    tree_ = std::make_unique<ART_ROWEX::Tree>(&load_key);
  }
  void destroy() {
    tree_.reset();
  }
  void thread_enter() {
    (void)current_threadinfo();
  }
  void thread_exit() {
    auto& state = current_thread_state();
    state.threadinfo.reset();
    state.tree = nullptr;
  }
  void insert(const key_slice_type* key, size_type key_length, value_type value, std::size_t tuple_id) {
    (void)value;
    auto tid = static_cast<TID>(tuple_id) + 1;
    Key art_key = make_key(key, key_length);
    tree_->insert(art_key, tid, current_threadinfo());
  }
  void erase(const key_slice_type* key, size_type key_length) {
    Key art_key = make_key(key, key_length);
    TID tid = tree_->lookup(art_key, current_threadinfo());
    if (tid != 0) {
      tree_->remove(art_key, tid, current_threadinfo());
    }
  }
  value_type find(const key_slice_type* key, size_type key_length) {
    Key art_key = make_key(key, key_length);
    TID tid = tree_->lookup(art_key, current_threadinfo());
    if (tid == 0) {
      return invalid_value;
    }
    return values_[tid - 1];
  }
  void scan(const key_slice_type* key, size_type key_length, uint32_t count, value_type* results) {
    Key start_key = make_key(key, key_length);
    Key end_key = make_upper_bound_key();
    Key continue_key;
    std::size_t num_results = 0;
    tree_->lookupRange(start_key, end_key, continue_key,
                       reinterpret_cast<TID*>(results), count, num_results,
                       current_threadinfo());
  }
  void print_stats() {}

 private:
  struct configs {
    std::size_t keylen_max; // parse again here; do not print
    configs() {}
    configs(std::vector<std::string>& arguments) {
      #define PARSE_DEFAULT_ARGUMENTS(arg, type, default_value) \
      [[maybe_unused]] auto tmp_##arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
      FORALL_ARGUMENTS(PARSE_DEFAULT_ARGUMENTS)
      #undef PARSE_DEFAULT_ARGUMENTS
      keylen_max = tmp_keylen_max;
      // ART does not support a key being prefix of another, so only support fixed length keys effectively
      check_argument(tmp_keylen_min == tmp_keylen_max);
    }
    void print() const {}
  };

  struct thread_state {
    const ART_ROWEX::Tree* tree = nullptr;
    std::unique_ptr<ART::ThreadInfo> threadinfo;
  };
  static thread_state& current_thread_state() {
    thread_local thread_state state;
    return state;
  }
  ART::ThreadInfo& current_threadinfo() {
    auto& state = current_thread_state();
    if (state.tree != tree_.get() || !state.threadinfo) {
      check_argument(tree_ != nullptr);
      state.threadinfo = std::make_unique<ART::ThreadInfo>(tree_->getThreadInfo());
      state.tree = tree_.get();
    }
    return *state.threadinfo;
  }

  static Key make_key(const key_slice_type* key, size_type key_length) {
    Key art_key;
    art_key.set(reinterpret_cast<const char*>(key), sizeof(key_slice_type) * key_length);
    return art_key;
  }
  static Key make_upper_bound_key() {
    Key upper_bound_key;
    const char upper_bound_byte = static_cast<char>(0xff);
    upper_bound_key.set(&upper_bound_byte, 1);
    return upper_bound_key;
  }
  static void load_key(TID tid, Key& key) {
    auto tuple_idx = static_cast<std::size_t>(tid - 1);
    key.set(reinterpret_cast<const char*>(&keys_[tuple_idx * key_stride_]), sizeof(key_slice_type) * key_lengths_[tuple_idx]);
  }

  configs configs_;
  std::unique_ptr<ART_ROWEX::Tree> tree_;
  static const key_slice_type* keys_;
  static const size_type* key_lengths_;
  static const value_type* values_;
  static size_type key_stride_;
};

const cpu_art_adapter::key_slice_type* cpu_art_adapter::keys_;
const cpu_art_adapter::size_type* cpu_art_adapter::key_lengths_;
const cpu_art_adapter::value_type* cpu_art_adapter::values_;
cpu_art_adapter::size_type cpu_art_adapter::key_stride_;
