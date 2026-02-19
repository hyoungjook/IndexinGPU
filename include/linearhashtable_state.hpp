/*
 *   Copyright 2025 Hyoungjoo Kim, Carnegie Mellon University
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
#include <macros.hpp>
#include <utils.hpp>

template <typename tile_type>
struct linearhashtable_state {
  using size_type = uint32_t;
  static constexpr int node_width = 32;
  static constexpr std::size_t bytes = node_width * sizeof(size_type);
  DEVICE_QUALIFIER linearhashtable_state(const tile_type& tile): tile_(tile) {}

  DEVICE_QUALIFIER void initialize(size_type initial_directory_size) {
    lane_elem_ = 0;
    if (tile_.thread_rank() == directory_size_lane_) {
      lane_elem_ = initial_directory_size;
    }
  }

  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load(size_type* ptr) {
    if constexpr (atomic) { tile_.sync(); }
    lane_elem_ = utils::memory::load<size_type, atomic, acquire>(ptr + tile_.thread_rank());
    if constexpr (atomic) { tile_.sync(); }
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store(size_type* ptr) {
    if constexpr (atomic) { tile_.sync(); }
    utils::memory::store<size_type, atomic, release>(ptr + tile_.thread_rank(), lane_elem_);
    if constexpr (atomic) { tile_.sync(); }
  }
  DEVICE_QUALIFIER void store_unlock(size_type* ptr) {
    if (tile_.thread_rank() == mutex_lane_) {
      lane_elem_ = 0;
    }
    store<true, true>(ptr);
  }

  template <bool atomic, bool acquire = true>
  static DEVICE_QUALIFIER size_type load_directory_size(size_type* ptr) {
    return utils::memory::load<size_type, atomic, acquire>(ptr + directory_size_lane_);
  }
  static DEVICE_QUALIFIER void increment_num_entries(size_type* ptr, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> num_entries_ref(ptr[num_entries_lane_]);
      num_entries_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
  }
  static DEVICE_QUALIFIER void decrement_num_entries(size_type* ptr, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> num_entries_ref(ptr[num_entries_lane_]);
      num_entries_ref.fetch_sub(1, cuda::memory_order_relaxed);
    }
  }

  DEVICE_QUALIFIER size_type get_directory_size() const {
    return tile_.shfl(lane_elem_, directory_size_lane_);
  }
  DEVICE_QUALIFIER void set_directory_size(size_type directory_size) {
    if (tile_.thread_rank() == directory_size_lane_) {
      lane_elem_ = directory_size;
    }
  }
  DEVICE_QUALIFIER size_type get_num_entries() const {
    return tile_.shfl(lane_elem_, num_entries_lane_);
  }

  DEVICE_QUALIFIER float get_load_factor() const {
    return static_cast<float>(get_num_entries()) / 15.0f / get_directory_size();
  }

  DEVICE_QUALIFIER bool is_locked() const {
    return tile_.shfl(lane_elem_, mutex_lane_);
  }
  DEVICE_QUALIFIER bool try_lock(size_type* ptr) {
    size_type old = 0;
    if (tile_.thread_rank() == mutex_lane_) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> mutex_ref(ptr[mutex_lane_]);
      mutex_ref.compare_exchange_strong(old, static_cast<size_type>(1),
                                        cuda::memory_order_acquire,
                                        cuda::memory_order_relaxed);
    }
    old = tile_.shfl(old, mutex_lane_);
    return (old == 0);
    // do not need to update registers; if locked, the code will load() again.
    // if lock failed, this node object will be disposed.
  }
  DEVICE_QUALIFIER void lock(size_type* ptr) {
    while (!try_lock(ptr));
    // the code will load() again
  }
  DEVICE_QUALIFIER void unlock(size_type* ptr) {
    assert(is_locked());
    if (tile_.thread_rank() == mutex_lane_) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> mutex_ref(ptr[mutex_lane_]);
      mutex_ref.store(0, cuda::memory_order_release);
      // the node object can be used after this, so update regsiters
      lane_elem_ = 0;
    }
  }

 private:
  size_type lane_elem_;
  const tile_type& tile_;

  // Each lane stores the following value:
  //  lane[0]: directory_size
  //  lane[1]: num_entries
  //  lane[2]: mutex
  static constexpr uint32_t directory_size_lane_ = 0;
  static constexpr uint32_t num_entries_lane_ = 1;
  static constexpr uint32_t mutex_lane_ = 2;

};
