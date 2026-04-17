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
#include <cuda_runtime.h>
#include <cstdint>
#include <macros.hpp>

namespace utils {

/*
 *    Fetches varlen key from global memory once and store in shared memory
 */
struct varlen_key_store {
  using key_slice_type = uint32_t;
  static constexpr uint32_t shmem_buffer_size = 16;
  template <typename tile_type>
  DEVICE_QUALIFIER varlen_key_store(key_slice_type* shmem_buffer, const key_slice_type* key, uint32_t key_length, const tile_type& tile)
      : shmem_buffer_(shmem_buffer), key_(key), current_slice_idx_(0) {
    static_assert(tile_type::size() >= shmem_buffer_size);
    tile.sync();
    if (tile.thread_rank() < key_length && tile.thread_rank() < shmem_buffer_size) {
      shmem_buffer_[tile.thread_rank()] = key_[tile.thread_rank()];
    }
    tile.sync();
  }
  DEVICE_QUALIFIER key_slice_type operator[](int idx) {
    int slice_idx = current_slice_idx_ + idx;
    assert(0 <= slice_idx);
    if (slice_idx < shmem_buffer_size) {
      return shmem_buffer_[slice_idx];
    }
    else {
      return key_[slice_idx];
    }
  }
  DEVICE_QUALIFIER varlen_key_store& operator+=(const int delta) {
    current_slice_idx_ += delta;
    return *this;
  }
  DEVICE_QUALIFIER varlen_key_store& operator-=(const int delta) {
    current_slice_idx_ -= delta;
    return *this;
  }
  DEVICE_QUALIFIER varlen_key_store operator+(const int delta) {
    return varlen_key_store(shmem_buffer_, key_, current_slice_idx_ + delta);
  }
  DEVICE_QUALIFIER varlen_key_store operator-(const int delta) {
    return varlen_key_store(shmem_buffer_, key_, current_slice_idx_ - delta);
  }

 private:
  DEVICE_QUALIFIER varlen_key_store(key_slice_type* shmem_buffer, const key_slice_type* key, int current_slice_idx)
    : shmem_buffer_(shmem_buffer), key_(key), current_slice_idx_(current_slice_idx) {}

  key_slice_type* const shmem_buffer_;
  const key_slice_type* const key_;
  int current_slice_idx_;
};


} // namespace utils
