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
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cmd.hpp>

#include <compiler.hh>
#include <kvthread.hh>
#include <masstree.hh>
#include <masstree_get.hh>
#include <masstree_insert.hh>
#include <masstree_remove.hh>
#include <masstree_scan.hh>
#include <value_string.hh>

struct cpu_masstree_adapter {
  static constexpr bool is_ordered = true;
  static constexpr bool support_mixed = true;
  static constexpr bool support_update = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  using table_value_type = value_string*;

  struct table_params : public Masstree::nodeparams<15, 15> {
    typedef table_value_type value_type;
    typedef Masstree::value_print<value_type> value_print_type;
    typedef threadinfo threadinfo_type;
  };

  using table_type = Masstree::basic_table<table_params>;
  using cursor_type = Masstree::tcursor<table_params>;
  using scan_key_type = Masstree::key<typename table_params::ikey_type>;
  using scan_stack_type = Masstree::scanstackelt<table_params>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void register_dataset(const key_slice_type* keys, const size_type* key_lengths,
                        const value_type* values, const size_type* value_lengths) {
    (void)keys;
    (void)key_lengths;
    (void)values;
    (void)value_lengths;
  }
  void initialize() {
    //check_argument(main_threadinfo_ == nullptr);
    main_threadinfo_ = threadinfo::make(threadinfo::TI_MAIN, -1);
    main_threadinfo_->pthread() = pthread_self();
    auto num_worker_threadinfos = std::max(1u, std::thread::hardware_concurrency());
    worker_threadinfos_.reserve(num_worker_threadinfos);
    operation_counts_.assign(num_worker_threadinfos, 0);
    for (unsigned thread_idx = 0; thread_idx < num_worker_threadinfos; thread_idx++) {
      worker_threadinfos_.push_back(threadinfo::make(threadinfo::TI_PROCESS, thread_idx));
    }
    table_ = std::make_unique<table_type>();
    table_->initialize(*main_threadinfo_);
  }
  void destroy() {
    if (table_) {
      destroy_visitor visitor;
      table_->scan(Masstree::Str(), true, visitor, *main_threadinfo_);
      table_->destroy(*main_threadinfo_);
      table_.reset();
    }
    if (main_threadinfo_) {
      drain_retired_nodes();
      release_threadinfos();
    }
  }

  void thread_enter(unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    ti.pthread() = pthread_self();
    ti.rcu_start();
  }
  void thread_exit(unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    ti.rcu_stop();
  }

  void insert(const key_slice_type* key, size_type key_length,
              const value_type* value, size_type value_length,
              std::size_t tuple_id, unsigned thread_idx) {
    (void)tuple_id;
    threadinfo& ti = get_threadinfo(thread_idx);
    cursor_type cursor(*table_, make_key(key, key_length));
    bool found = cursor.find_insert(ti);
    table_value_type old_value = found ? cursor.value() : nullptr;
    cursor.value() = make_value(value, value_length, ti);
    fence();
    cursor.finish(found ? 0 : 1, ti);
    if (old_value) {
      old_value->deallocate_rcu(ti);
    }
    maybe_quiesce(ti, thread_idx);
  }
  void update(const key_slice_type* key, size_type key_length,
              const value_type* value, size_type value_length,
              std::size_t tuple_id, unsigned thread_idx) {
    (void)tuple_id;
    threadinfo& ti = get_threadinfo(thread_idx);
    cursor_type cursor(*table_, make_key(key, key_length));
    bool found = cursor.find_insert(ti);
    table_value_type old_value = found ? cursor.value() : nullptr;
    cursor.value() = make_value(value, value_length, ti);
    fence();
    cursor.finish(found ? 0 : 1, ti);
    if (old_value) {
      old_value->deallocate_rcu(ti);
    }
    maybe_quiesce(ti, thread_idx);
  }
  void erase(const key_slice_type* key, size_type key_length, unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    cursor_type cursor(*table_, make_key(key, key_length));
    bool found = cursor.find_locked(ti);
    table_value_type old_value = found ? cursor.value() : nullptr;
    cursor.finish(found ? -1 : 0, ti);
    if (old_value) {
      old_value->deallocate_rcu(ti);
    }
    maybe_quiesce(ti, thread_idx);
  }
  void find(const key_slice_type* key, size_type key_length,
            value_type* result, size_type* result_length,
            unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    table_value_type value = nullptr;
    if (table_->get(make_key(key, key_length), value, ti)) {
      auto string = value->col(0);
      std::memcpy(result, string.data(), string.length());
      *result_length = string.length() / sizeof(value_type);
    } else {
      result[0] = invalid_value;
      *result_length = 0;
    }
    maybe_quiesce(ti, thread_idx);
  }
  void scan(const key_slice_type* key, size_type key_length, uint32_t count,
            value_type* results, size_type value_stride,
            size_type* result_lengths, unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    scan_visitor visitor(count, results, value_stride, result_lengths);
    table_->scan(make_key(key, key_length), true, visitor, ti);
    for (uint32_t i = visitor.num_results; i < count; i++) {
      results[i * value_stride] = invalid_value;
      result_lengths[i] = 0;
    }
    maybe_quiesce(ti, thread_idx);
  }
  void print_stats() {}
  void ht_print_load_factor(std::size_t max_keys, uint32_t key_length, uint32_t value_length) {
    (void)max_keys;
    (void)key_length;
    (void)value_length;
  }

 private:
  struct configs {
    configs() = default;
    explicit configs(std::vector<std::string>& arguments) {
      auto keylen_min = get_arg_value<size_type>(arguments, "keylen_min").value_or(1);
      auto keylen_max = get_arg_value<size_type>(arguments, "keylen_max").value_or(1);
      auto valuelen_min = get_arg_value<size_type>(arguments, "valuelen_min").value_or(1);
      auto valuelen_max = get_arg_value<size_type>(arguments, "valuelen_max").value_or(1);
      check_argument(valuelen_min == valuelen_max);
      check_argument(valuelen_max == 1 || valuelen_max == 2 || valuelen_max == 4 ||
                     valuelen_max == 8 || valuelen_max == 16);
      check_argument(valuelen_max == 1 || (keylen_min == 1 && keylen_max == 1));
    }
    void print() const {
    }
  };

  struct scan_visitor {
    uint32_t limit;
    value_type* results;
    size_type value_stride;
    size_type* result_lengths;
    uint32_t num_results = 0;

    scan_visitor(uint32_t limit, value_type* results, size_type value_stride,
                 size_type* result_lengths)
        : limit(limit), results(results), value_stride(value_stride),
          result_lengths(result_lengths) {
    }

    template <typename Stack, typename Key>
    void visit_leaf(const Stack&, const Key&, threadinfo&) {
    }

    bool visit_value(Masstree::Str, table_value_type value, threadinfo&) {
      auto string = value->col(0);
      std::memcpy(results + num_results * value_stride, string.data(), string.length());
      result_lengths[num_results++] = string.length() / sizeof(value_type);
      return num_results < limit;
    }
  };

  struct destroy_visitor {
    template <typename Stack, typename Key>
    void visit_leaf(const Stack&, const Key&, threadinfo&) {
    }

    bool visit_value(Masstree::Str, table_value_type value, threadinfo& ti) {
      value->deallocate(ti);
      return true;
    }
  };

  static Masstree::Str make_key(const key_slice_type* key, size_type key_length) {
    return Masstree::Str(reinterpret_cast<const char*>(key),
                         static_cast<int>(key_length * sizeof(key_slice_type)));
  }

  static table_value_type make_value(const value_type* value,
                                     size_type value_length, threadinfo& ti) {
    return value_string::create1(
        Masstree::Str(reinterpret_cast<const char*>(value),
                      value_length * sizeof(value_type)),
        0, ti);
  }

  threadinfo& get_threadinfo(unsigned thread_idx) {
    check_argument(thread_idx < worker_threadinfos_.size());
    return *worker_threadinfos_[thread_idx];
  }

  void advance_global_epoch() {
    std::lock_guard<std::mutex> lock(epoch_mutex_);
    globalepoch.store(globalepoch.load() + 2);
    active_epoch.store(threadinfo::min_active_epoch());
  }

  void maybe_quiesce(threadinfo& ti, unsigned thread_idx) {
    if ((++operation_counts_[thread_idx] & 63) == 0) {
      advance_global_epoch();
      ti.rcu_quiesce();
    }
  }

  void drain_threadinfo(threadinfo& ti) {
    while (ti.has_pending_rcu()) {
      advance_global_epoch();
      ti.rcu_quiesce();
    }
  }

  void drain_retired_nodes() {
    if (main_threadinfo_) {
      drain_threadinfo(*main_threadinfo_);
    }
    for (auto* ti: worker_threadinfos_) {
      if (ti) {
        drain_threadinfo(*ti);
      }
    }
  }

  void release_threadinfos() {
    for (auto*& ti: worker_threadinfos_) {
      if (ti) {
        threadinfo::destroy(ti);
        ti = nullptr;
      }
    }
    worker_threadinfos_.clear();
    operation_counts_.clear();
    threadinfo::destroy(main_threadinfo_);
    main_threadinfo_ = nullptr;
  }

  configs configs_;
  std::unique_ptr<table_type> table_;
  threadinfo* main_threadinfo_ = nullptr;
  std::vector<threadinfo*> worker_threadinfos_;
  std::vector<uint32_t> operation_counts_;
  std::mutex epoch_mutex_;
};
