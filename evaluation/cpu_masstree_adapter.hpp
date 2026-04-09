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

struct cpu_masstree_adapter {
  static constexpr bool is_ordered = true;
  static constexpr bool support_mixed = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  using table_value_type = value_type;

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
  void register_dataset(const key_slice_type* key, const size_type* key_lengths, const value_type* values) {}
  void initialize() {
    check_argument(main_threadinfo_ == nullptr);
    main_threadinfo_ = threadinfo::make(threadinfo::TI_MAIN, -1);
    main_threadinfo_->pthread() = pthread_self();
    auto num_worker_threadinfos = std::max(1u, std::thread::hardware_concurrency());
    worker_threadinfos_.reserve(num_worker_threadinfos);
    for (unsigned thread_idx = 0; thread_idx < num_worker_threadinfos; thread_idx++) {
      worker_threadinfos_.push_back(threadinfo::make(threadinfo::TI_PROCESS, thread_idx));
    }
    table_ = std::make_unique<table_type>();
    table_->initialize(*main_threadinfo_);
  }
  void destroy() {
    if (table_) {
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

  void insert(const key_slice_type* key, size_type key_length, value_type value, std::size_t tuple_id, unsigned thread_idx) {
    (void)tuple_id;
    threadinfo& ti = get_threadinfo(thread_idx);
    cursor_type cursor(*table_, make_key(key, key_length));
    cursor.find_insert(ti);
    cursor.value() = value;
    fence();
    cursor.finish(1, ti);
  }
  void erase(const key_slice_type* key, size_type key_length, unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    cursor_type cursor(*table_, make_key(key, key_length));
    bool found = cursor.find_locked(ti);
    cursor.finish(found ? -1 : 0, ti);
  }
  value_type find(const key_slice_type* key, size_type key_length, unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    value_type value = invalid_value;
    table_->get(make_key(key, key_length), value, ti);
    return value;
  }
  void scan(const key_slice_type* key, size_type key_length, uint32_t count, value_type* results, unsigned thread_idx) {
    threadinfo& ti = get_threadinfo(thread_idx);
    scan_visitor visitor(count, results);
    table_->scan(make_key(key, key_length), true, visitor, ti);
    std::fill(results + visitor.num_results, results + count, invalid_value);
  }
  void print_stats() {}

 private:
  struct configs {
    configs() = default;
    explicit configs(std::vector<std::string>& arguments) {
    }
    void print() const {
    }
  };

  struct scan_visitor {
    uint32_t limit;
    value_type* results;
    uint32_t num_results = 0;

    scan_visitor(uint32_t limit, value_type* results)
        : limit(limit), results(results) {
    }

    template <typename Stack, typename Key>
    void visit_leaf(const Stack&, const Key&, threadinfo&) {
    }

    bool visit_value(Masstree::Str, value_type value, threadinfo&) {
      results[num_results++] = value;
      return num_results < limit;
    }
  };

  static Masstree::Str make_key(const key_slice_type* key, size_type key_length) {
    return Masstree::Str(reinterpret_cast<const char*>(key),
                         static_cast<int>(key_length * sizeof(key_slice_type)));
  }

  threadinfo& get_threadinfo(unsigned thread_idx) {
    check_argument(thread_idx < worker_threadinfos_.size());
    return *worker_threadinfos_[thread_idx];
  }

  static void advance_global_epoch() {
    globalepoch.store(globalepoch.load() + 2);
    active_epoch.store(threadinfo::min_active_epoch());
  }

  static void drain_threadinfo(threadinfo& ti) {
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
    threadinfo::destroy(main_threadinfo_);
    main_threadinfo_ = nullptr;
  }

  configs configs_;
  std::unique_ptr<table_type> table_;
  threadinfo* main_threadinfo_ = nullptr;
  std::vector<threadinfo*> worker_threadinfos_;
};
