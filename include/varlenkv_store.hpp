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
namespace varlenkv {
using slice_type = uint32_t;
inline const uint32_t shmem_buffer_size = 16;

template <typename reg_t, typename tile_t>
DEVICE_QUALIFIER reg_t shfl_reg(reg_t value, int rank, const tile_t& tile) {
  auto shfl_value = tile.shfl(*reinterpret_cast<uint64_t*>(&value), rank);
  return *reinterpret_cast<reg_t*>(&shfl_value);
}

union reg_const_type {
  const slice_type* ptr;
  slice_type raw;
};

DEVICE_QUALIFIER reg_const_type init_reg_const(const slice_type* ptr, uint32_t thread_id, uint32_t max_key_length) {
  reg_const_type reg;
  if (max_key_length == 1) {
    reg.raw = ptr[thread_id];
  }
  else {
    reg.ptr = &ptr[max_key_length * thread_id];
  }
  return reg;
}

template <bool use_shmem_key>
struct wrapper_const {
  template <typename tile_type>
  DEVICE_QUALIFIER wrapper_const(const reg_const_type& reg,
                                 uint32_t key_length,
                                 uint32_t max_key_length,
                                 slice_type* shmem_buffer,
                                 const tile_type& tile)
      : key_info_reg_(reg)
      , current_slice_idx_(0)
      , max_key_length_(max_key_length)
      , shmem_buffer_(shmem_buffer) {
    static_assert(!use_shmem_key || tile_type::size() >= shmem_buffer_size);
    if constexpr (use_shmem_key) {
      tile.sync();
      if (max_key_length_ == 1) {
        if (tile.thread_rank() == 0) {
          shmem_buffer_[0] = key_info_reg_.raw;
        }
      }
      else {
        if (tile.thread_rank() < min(key_length, shmem_buffer_size)) {
          shmem_buffer_[tile.thread_rank()] = key_info_reg_.ptr[tile.thread_rank()];
        }
      }
      tile.sync();
    }
  }
  DEVICE_QUALIFIER slice_type operator[](int idx) {
    int slice_idx = current_slice_idx_ + idx;
    assert(slice_idx >= 0);
    if constexpr (use_shmem_key) {
      return (slice_idx < shmem_buffer_size) ?
        shmem_buffer_[slice_idx] :
        key_info_reg_.ptr[slice_idx];
    }
    else {
      if (max_key_length_ == 1) {
        return key_info_reg_.raw;
      }
      else {
        return key_info_reg_.ptr[slice_idx];
      }
    }
  }
  DEVICE_QUALIFIER wrapper_const operator+(const int delta) {
    return wrapper_const(*this, delta);
  }
  DEVICE_QUALIFIER wrapper_const operator-(const int delta) {
    return wrapper_const(*this, -delta);
  }

 private:
  DEVICE_QUALIFIER wrapper_const(const wrapper_const& other,
                                 const int delta)
    : key_info_reg_(other.key_info_reg_)
    , current_slice_idx_(other.current_slice_idx_ + delta)
    , max_key_length_(other.max_key_length_)
    , shmem_buffer_(other.shmem_buffer_) {}
  const reg_const_type& key_info_reg_;
  const int current_slice_idx_;
  const uint32_t max_key_length_;
  slice_type* const shmem_buffer_;
};

union reg_type {
  slice_type* ptr;
  slice_type raw;
};

DEVICE_QUALIFIER reg_type init_reg(slice_type* ptr, uint32_t thread_id, uint32_t max_key_length) {
  reg_type reg;
  if (max_key_length == 1) {
    reg.raw = ptr[thread_id];
  }
  else {
    reg.ptr = &ptr[max_key_length * thread_id];
  }
  return reg;
}

template <bool use_shmem_key>
struct wrapper {
  template <typename tile_type>
  DEVICE_QUALIFIER wrapper(reg_type& reg,
                           uint32_t key_length,
                           uint32_t max_key_length,
                           slice_type* shmem_buffer,
                           const tile_type& tile)
      : key_info_reg_(reg)
      , current_slice_idx_(0)
      , max_key_length_(max_key_length)
      , shmem_buffer_(shmem_buffer) {
    static_assert(!use_shmem_key || tile_type::size() >= shmem_buffer_size);
    if constexpr (use_shmem_key) {
      tile.sync();
      if (max_key_length_ == 1) {
        if (tile.thread_rank() == 0) {
          shmem_buffer_[0] = key_info_reg_.raw;
        }
      }
      else {
        if (tile.thread_rank() < min(key_length, shmem_buffer_size)) {
          shmem_buffer_[tile.thread_rank()] = key_info_reg_.ptr[tile.thread_rank()];
        }
      }
      tile.sync();
    }
  }
  DEVICE_QUALIFIER slice_type& operator[](int idx) {
    int slice_idx = current_slice_idx_ + idx;
    assert(slice_idx >= 0);
    if constexpr (use_shmem_key) {
      return (slice_idx < shmem_buffer_size) ?
        shmem_buffer_[slice_idx] :
        key_info_reg_.ptr[slice_idx];
    }
    else {
      if (max_key_length_ == 1) {
        return key_info_reg_.raw;
      }
      else {
        return key_info_reg_.ptr[slice_idx];
      }
    }
  }
  DEVICE_QUALIFIER wrapper operator+(const int delta) {
    return wrapper(*this, delta);
  }
  DEVICE_QUALIFIER wrapper operator-(const int delta) {
    return wrapper(*this, -delta);
  }

 private:
  DEVICE_QUALIFIER wrapper(const wrapper& other,
                           const int delta)
    : key_info_reg_(other.key_info_reg_)
    , current_slice_idx_(other.current_slice_idx_ + delta)
    , max_key_length_(other.max_key_length_)
    , shmem_buffer_(other.shmem_buffer_) {}
  reg_type& key_info_reg_;
  const int current_slice_idx_;
  const uint32_t max_key_length_;
  slice_type* const shmem_buffer_;
};


} // namespace varlenkv
} // namespace utils
