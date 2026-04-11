/*
 *   Copyright 2022 The Regents of the University of California, Davis
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

namespace utils {
namespace bits {
// Bit Field Extract.
DEVICE_QUALIFIER int bfe(uint32_t src, int num_bits) {
  unsigned mask;
  asm("bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits));
  return mask;
}

// Find most significant non - sign bit.
// bfind(0) = -1, bfind(1) = 0
DEVICE_QUALIFIER int bfind(uint32_t src) {
  int msb;
  asm("bfind.u32 %0, %1;" : "=r"(msb) : "r"(src));
  return msb;
}
DEVICE_QUALIFIER int bfind(uint64_t src) {
  int msb;
  asm("bfind.u64 %0, %1;" : "=r"(msb) : "l"(src));
  return msb;
}
};  // namespace bits

struct device_memory_usage_results {
  std::size_t used_bytes;
  std::size_t total_bytes;
};
device_memory_usage_results compute_device_memory_usage() {
  std::size_t total_bytes;
  std::size_t free_bytes;
  cuda_try(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::size_t used_bytes = total_bytes - free_bytes;
  return {used_bytes, free_bytes};
}
void set_cuda_buffer_size(const std::size_t new_size, const cudaLimit limit) {
  cuda_try(cudaDeviceSetLimit(limit, new_size));
}

std::size_t get_cuda_buffer_size(const cudaLimit limit) {
  std::size_t cur_size;
  cuda_try(cudaDeviceGetLimit(&cur_size, limit));
  return cur_size;
}

__host__ __device__ static constexpr uint32_t constexpr_pow(uint32_t base, uint32_t exp) {
  return (exp == 0) ? 1 : base * constexpr_pow(base, exp - 1);
}

namespace memory {

template <typename T, bool atomic, bool acquire = false>
DEVICE_QUALIFIER T load(T* ptr) {
  if constexpr (atomic) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ptr_ref(*ptr);
    if constexpr (acquire) {
      return ptr_ref.load(cuda::memory_order_acquire);
    }
    else {
      return ptr_ref.load(cuda::memory_order_relaxed);
    }
  }
  else {
    return *ptr;
  }
}

template <typename T, bool atomic, bool release = false>
DEVICE_QUALIFIER void store(T* ptr, T value) {
  if constexpr (atomic) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ptr_ref(*ptr);
    if constexpr (release) {
      ptr_ref.store(value, cuda::memory_order_release);
    }
    else {
      ptr_ref.store(value, cuda::memory_order_relaxed);
    }
  }
  else {
    *ptr = value;
  }
}

}; // namespace memory

namespace tile {

// Lightweight tile that replaces cooperative_groups::tiled_partition
struct full_warp_tile {
  static constexpr uint32_t size() { return size_; }
  DEVICE_QUALIFIER full_warp_tile() {}
  DEVICE_QUALIFIER uint32_t thread_rank() const {
    return static_cast<uint32_t>(threadIdx.x) & 31u;
  }
  DEVICE_QUALIFIER void sync() const {
    __syncwarp();
  }
  DEVICE_QUALIFIER uint32_t ballot(int predicate) const {
    return __ballot_sync(mask_, predicate);
  }
  DEVICE_QUALIFIER int all(int predicate) const {
    return __ballot_sync(mask_, predicate) == mask_;
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl(T elem, int srcRank) const {
    if constexpr (std::is_pointer_v<T>) {
      return reinterpret_cast<Tret>(__shfl_sync(mask_, reinterpret_cast<uintptr_t>(elem), srcRank, size_));
    }
    else {
      return static_cast<Tret>(__shfl_sync(mask_, elem, srcRank, size_));
    }
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl_down(T elem, unsigned int delta) const {
    return __shfl_down_sync(mask_, elem, delta, size_);
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl_up(T elem, unsigned int delta) const {
    return __shfl_up_sync(mask_, elem, delta, size_);
  }
private:
  static constexpr int size_ = 32;
  static constexpr uint32_t mask_ = 0xffffffffu;
};

struct half_warp_tile {
  static constexpr uint32_t size() { return size_; }
  DEVICE_QUALIFIER half_warp_tile() {
    bool is_second_half = (static_cast<uint32_t>(threadIdx.x) & 16u);
    mask_ = is_second_half ? second_half_mask_ : first_half_mask_;
  }
  DEVICE_QUALIFIER uint32_t thread_rank() const {
    return static_cast<uint32_t>(threadIdx.x) & 15u;
  }
  DEVICE_QUALIFIER void sync() const {
    __syncwarp(mask_);
  }
  DEVICE_QUALIFIER uint32_t ballot(int predicate) const {
    // ballot then shift by (mask_ == first_half_mask_ ? 0 : 16)
    return __ballot_sync(mask_, predicate) >> ((~mask_) & 16u);
  }
  DEVICE_QUALIFIER int all(int predicate) const {
    return __ballot_sync(mask_, predicate) == mask_;
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl(T elem, int srcRank) const {
    if constexpr (std::is_pointer_v<T>) {
      return reinterpret_cast<Tret>(__shfl_sync(mask_, reinterpret_cast<uintptr_t>(elem), srcRank, size_));
    }
    else {
      return static_cast<Tret>(__shfl_sync(mask_, elem, srcRank, size_));
    }
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl_down(T elem, unsigned int delta) const {
    return __shfl_down_sync(mask_, elem, delta, size_);
  }
  template <typename T, typename Tret = std::remove_const_t<T>>
  DEVICE_QUALIFIER Tret shfl_up(T elem, unsigned int delta) const {
    return __shfl_up_sync(mask_, elem, delta, size_);
  }
private:
  uint32_t mask_;
  static constexpr int size_ = 16;
  static constexpr uint32_t first_half_mask_ = 0x0000ffffu;
  static constexpr uint32_t second_half_mask_ = 0xffff0000u;
};

template <uint32_t tile_size>
struct lightweight_tiled_partition_s {
  static_assert(tile_size == 32 || tile_size == 16);
  using type = std::conditional_t<tile_size == 32,
                                  full_warp_tile,
                                  half_warp_tile>;
};

template <uint32_t tile_size>
using lightweight_tiled_partition = typename lightweight_tiled_partition_s<tile_size>::type;

}; // namespace tile

};  // namespace utils
