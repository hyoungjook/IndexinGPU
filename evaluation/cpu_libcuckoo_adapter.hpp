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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <libcuckoo/cuckoohash_map.hh>

struct cpu_libcuckoo_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using key_type = std::vector<key_slice_type>;

  struct key_hash {
    std::size_t operator()(const key_type& key) const {
      std::size_t hash = 0;
      for (auto slice : key) {
        hash ^= std::hash<key_slice_type>{}(slice) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  using index_type = libcuckoo::cuckoohash_map<key_type, value_type, key_hash>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    index_ = std::make_unique<index_type>(configs_.initial_capacity);
  }
  void destroy() {
    index_.reset();
  }
  void insert(const key_slice_type* key, size_type key_length, value_type value) {
    index_->insert_or_assign(make_key(key, key_length), value);
  }
  void erase(const key_slice_type* key, size_type key_length) {
    index_->erase(make_key(key, key_length));
  }
  value_type find(const key_slice_type* key, size_type key_length) {
    value_type value = std::numeric_limits<value_type>::max();
    index_->find(make_key(key, key_length), value);
    return value;
  }

 private:
  struct configs {
    std::size_t initial_capacity;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      initial_capacity = get_arg_value<float>(arguments, "initial-capacity").value_or(100000);
      check_argument(0 < initial_capacity);
    }
    void print() const {
      std::cout << "  initial-capacity: " << initial_capacity << std::endl;
    }
  };

  configs configs_;
  std::unique_ptr<index_type> index_;

  static key_type make_key(const key_slice_type* key, size_type key_length) {
    return key_type(key, key + key_length);
  }
};
