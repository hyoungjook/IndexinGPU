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
    if (!main_threadinfo_) {
      main_threadinfo_ = threadinfo::make(threadinfo::TI_MAIN, -1);
    }
    table_ = std::make_unique<table_type>();
    table_->initialize(*main_threadinfo_);
  }
  void destroy() {
    if (table_) {
      table_->destroy(*main_threadinfo_);
      table_.reset();
    }
  }

  void thread_enter() {
    threadinfo& ti = current_threadinfo();
    ti.rcu_start();
  }
  void thread_exit() {
    threadinfo& ti = current_threadinfo();
    ti.rcu_stop();
  }

  void insert(const key_slice_type* key, size_type key_length, value_type value) {
    threadinfo& ti = current_threadinfo();
    cursor_type cursor(*table_, make_key(key, key_length));
    cursor.find_insert(ti);
    cursor.value() = value;
    fence();
    cursor.finish(1, ti);
  }
  void erase(const key_slice_type* key, size_type key_length) {
    threadinfo& ti = current_threadinfo();
    cursor_type cursor(*table_, make_key(key, key_length));
    bool found = cursor.find_locked(ti);
    cursor.finish(found ? -1 : 0, ti);
  }
  value_type find(const key_slice_type* key, size_type key_length) {
    threadinfo& ti = current_threadinfo();
    value_type value = invalid_value;
    table_->get(make_key(key, key_length), value, ti);
    return value;
  }
  void scan(const key_slice_type* key, size_type key_length, uint32_t count, value_type* results) {
    threadinfo& ti = current_threadinfo();
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

  static threadinfo& current_threadinfo() {
    thread_local threadinfo* threadinfo_ptr = nullptr;
    if (!threadinfo_ptr) {
      threadinfo_ptr = threadinfo::make(threadinfo::TI_PROCESS, next_thread_id_.fetch_add(1));
    }
    return *threadinfo_ptr;
  }

  configs configs_;
  std::unique_ptr<table_type> table_;
  threadinfo* main_threadinfo_ = nullptr;

  inline static std::atomic<int> next_thread_id_{0};
};
