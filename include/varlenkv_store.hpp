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
  return reg_t{.ptr = tile.shfl(value.ptr, rank)};
}

union reg_input_type {
  const slice_type* ptr;
  slice_type raw;
};

DEVICE_QUALIFIER reg_input_type init_reg_input(const slice_type* ptr, uint32_t thread_id, uint32_t max_length) {
  reg_input_type reg;
  if (max_length == 1) {
    reg.raw = ptr[thread_id];
  }
  else {
    reg.ptr = &ptr[max_length * thread_id];
  }
  return reg;
}

template <bool use_shmem_key>
struct wrapper_input {
  template <typename tile_type>
  DEVICE_QUALIFIER wrapper_input(const reg_input_type& reg,
                                 uint32_t length,
                                 uint32_t max_length,
                                 slice_type* shmem_buffer,
                                 const tile_type& tile)
      : reg_(reg)
      , current_slice_idx_(0)
      , max_length_(max_length)
      , shmem_buffer_(shmem_buffer) {
    static_assert(!use_shmem_key || tile_type::size() >= shmem_buffer_size);
    if constexpr (use_shmem_key) {
      if (max_length_ == 1) {
        if (tile.thread_rank() == 0) {
          shmem_buffer_[0] = reg_.raw;
        }
      }
      else {
        if (tile.thread_rank() < min(length, shmem_buffer_size)) {
          shmem_buffer_[tile.thread_rank()] = reg_.ptr[tile.thread_rank()];
        }
      }
      tile.sync();
    }
  }
  DEVICE_QUALIFIER slice_type operator[](int idx) const {
    int slice_idx = current_slice_idx_ + idx;
    assert(slice_idx >= 0);
    if constexpr (use_shmem_key) {
      return (slice_idx < shmem_buffer_size) ?
        shmem_buffer_[slice_idx] :
        reg_.ptr[slice_idx];
    }
    else {
      if (max_length_ == 1) {
        return reg_.raw;
      }
      else {
        return reg_.ptr[slice_idx];
      }
    }
  }
  DEVICE_QUALIFIER wrapper_input operator+(const int delta) {
    return wrapper_input(*this, delta);
  }
  DEVICE_QUALIFIER wrapper_input operator-(const int delta) {
    return wrapper_input(*this, -delta);
  }

 private:
  DEVICE_QUALIFIER wrapper_input(const wrapper_input& other,
                                 const int delta)
    : reg_(other.reg_)
    , current_slice_idx_(other.current_slice_idx_ + delta)
    , max_length_(other.max_length_)
    , shmem_buffer_(other.shmem_buffer_) {}
  const reg_input_type& reg_;
  const int current_slice_idx_;
  const uint32_t max_length_;
  slice_type* const shmem_buffer_;
};

union reg_output_type {
  slice_type* ptr;
  slice_type raw;
};

template <bool use_shmem_key>
DEVICE_QUALIFIER void fini_reg_output(reg_output_type& reg, slice_type* ptr, uint32_t thread_id, uint32_t max_length) {
  if constexpr (use_shmem_key) {
    if (max_length == 1) {
      ptr[thread_id] = reg.raw;
    }
  }
}

template <bool use_shmem_key>
struct wrapper_output {
  DEVICE_QUALIFIER wrapper_output(const reg_output_type& reg,
                                  uint32_t max_length,
                                  slice_type* shmem_buffer)
      : reg_(reg)
      , current_slice_idx_(0)
      , max_length_(max_length)
      , shmem_buffer_(shmem_buffer) {}
  template <typename tile_type>
  DEVICE_QUALIFIER void flush(uint32_t length, const tile_type& tile, reg_output_type& perlane_reg, int cur_rank) {
    static_assert(!use_shmem_key || tile_type::size() >= shmem_buffer_size);
    if constexpr (use_shmem_key) {
      tile.sync();
      if (max_length_ == 1) {
        // stores slice to cur_rank's register
        if (tile.thread_rank() == cur_rank) {
          perlane_reg.raw = shmem_buffer_[0];
        }
      }
      else {
        // flush to global memory
        if (tile.thread_rank() < min(length, shmem_buffer_size)) {
          reg_.ptr[tile.thread_rank()] = shmem_buffer_[tile.thread_rank()];
        }
      }
    }
  }

  DEVICE_QUALIFIER slice_type& operator[](int idx) {
    int slice_idx = current_slice_idx_ + idx;
    assert(slice_idx >= 0);
    if constexpr (use_shmem_key) {
      return (slice_idx < shmem_buffer_size) ?
        shmem_buffer_[slice_idx] :
        reg_.ptr[slice_idx];
    }
    else {
      return reg_.ptr[slice_idx];
    }
  }
  DEVICE_QUALIFIER wrapper_output operator+(const int delta) {
    return wrapper_output(*this, delta);
  }
  DEVICE_QUALIFIER wrapper_output operator-(const int delta) {
    return wrapper_output(*this, -delta);
  }

 private:
  DEVICE_QUALIFIER wrapper_output(const wrapper_output& other,
                           const int delta)
    : reg_(other.reg_)
    , current_slice_idx_(other.current_slice_idx_ + delta)
    , max_length_(other.max_length_)
    , shmem_buffer_(other.shmem_buffer_) {}
  const reg_output_type& reg_;
  const int current_slice_idx_;
  const uint32_t max_length_;
  slice_type* const shmem_buffer_;
};

union reg_input_or_output_type {
  reg_input_type input;
  reg_output_type output;
};

} // namespace varlenkv
} // namespace utils
