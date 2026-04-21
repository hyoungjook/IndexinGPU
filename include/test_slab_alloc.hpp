/*
 *   Copyright 2022 The Regents of the University of California, Davis
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

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <device_context.hpp>
#include <macros.hpp>
#include <utils.hpp>

namespace test_slab_allocator_detail {

template <typename T>
constexpr bool is_pow2(T x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

constexpr uint32_t floor_pow2_u32(uint32_t x) {
  if (x == 0) { return 0; }
  uint32_t result = 1;
  while (result <= (x >> 1)) {
    result <<= 1;
  }
  return result;
}

constexpr uint32_t pow2_log2_u32(uint32_t x) {
  uint32_t result = 0;
  while (x > 1) {
    x >>= 1;
    result++;
  }
  return result;
}

template <typename T>
DEVICE_QUALIFIER int cuda_ffs(const T& x) {
  return __ffsll(static_cast<unsigned long long>(x));
}

DEVICE_QUALIFIER int cuda_ffs(const uint32_t& x) {
  return __ffs(x);
}

}  // namespace test_slab_allocator_detail

template <uint32_t slab_size = 128>
struct test_slab_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;

  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint32_t bitmap_header_bytes_ = 32 * sizeof(uint32_t);
  static constexpr uint32_t num_slabs_in_block_ = bitmap_header_bytes_ * 8;
  static constexpr uint32_t block_address_bits_ = 10;
  static constexpr uint32_t block_size_ = (num_slabs_in_block_ + 1) * slab_size_;

  static_assert(num_slabs_in_block_ == 1024);
  static_assert(slab_size_ == bitmap_header_bytes_,
                "test_slab_allocator currently assumes 128B slabs");

  struct device_instance_type {
    void* pool_;
    pointer_type num_superblocks_;
    pointer_type num_memory_blocks_;
    pointer_type memory_block_mask_;
    uint32_t superblock_shift_;
  };

  explicit test_slab_allocator(float pool_ratio = 0.9f) {
    auto meminfo = utils::compute_device_memory_usage();
    auto max_bytes =
        static_cast<std::size_t>(static_cast<double>(meminfo.total_bytes) * pool_ratio);
    auto max_blocks_by_bytes =
        static_cast<uint32_t>(max_bytes / static_cast<std::size_t>(block_size_));
    total_blocks_ = test_slab_allocator_detail::floor_pow2_u32(max_blocks_by_bytes);
    if (total_blocks_ == 0) {
      std::cerr << "test_slab_allocator: pool too small for one allocation block"
                << std::endl;
      std::abort();
    }
    if (static_cast<std::size_t>(total_blocks_) * num_slabs_in_block_ >=
        std::numeric_limits<pointer_type>::max()) {
      std::cerr << "test_slab_allocator: pointer exceeds uint32 limit" << std::endl;
      std::abort();
    }

    int device = 0;
    cuda_try(cudaGetDevice(&device));
    cudaDeviceProp prop;
    cuda_try(cudaGetDeviceProperties(&prop, device));
    auto sm_pow2 = test_slab_allocator_detail::floor_pow2_u32(
        static_cast<uint32_t>(std::max(1, prop.multiProcessorCount)));
    num_superblocks_ = std::min(total_blocks_, std::max<uint32_t>(1, sm_pow2));
    if (num_superblocks_ == 0) {
      num_superblocks_ = 1;
    }
    num_memory_blocks_ = total_blocks_ / num_superblocks_;
    if (!test_slab_allocator_detail::is_pow2(num_superblocks_) ||
        !test_slab_allocator_detail::is_pow2(num_memory_blocks_)) {
      std::cerr << "test_slab_allocator: internal block geometry must stay power-of-two"
                << std::endl;
      std::abort();
    }
    superblock_shift_ =
        block_address_bits_ +
        test_slab_allocator_detail::pow2_log2_u32(num_memory_blocks_);

    auto total_bytes = static_cast<std::size_t>(total_blocks_) * block_size_;
    cuda_try(cudaMalloc(&pool_, total_bytes));
    cuda_try(cudaMemset(pool_, 0x00, total_bytes));
  }

  ~test_slab_allocator() {
    cuda_try(cudaFree(pool_));
  }

  test_slab_allocator(const test_slab_allocator& other) = delete;
  test_slab_allocator& operator=(const test_slab_allocator& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{
        pool_,
        num_superblocks_,
        num_memory_blocks_,
        num_memory_blocks_ - 1,
        superblock_shift_};
  }

  void print_stats() const {
    std::vector<uint32_t> h_bitmap(
        static_cast<std::size_t>(total_blocks_) * bitmap_header_bytes_ / sizeof(uint32_t));
    cuda_try(cudaMemcpy2D(
        h_bitmap.data(),
        bitmap_header_bytes_,
        pool_,
        block_size_,
        bitmap_header_bytes_,
        total_blocks_,
        cudaMemcpyDeviceToHost));

    std::size_t slab_count = 0;
    for (uint32_t word : h_bitmap) {
      slab_count += __builtin_popcount(word);
    }
    const uint64_t total_slabs =
        static_cast<uint64_t>(total_blocks_) * num_slabs_in_block_;
    std::cout << "test_slab_allocator(" << slab_size_ << "B slabs, "
              << num_superblocks_ << "x" << num_memory_blocks_ << " blocks): "
              << slab_count << "/" << total_slabs << " slabs allocated "
              << "(" << (static_cast<float>(slab_count) / total_slabs) * 100.0f << "%)"
              << std::endl;
  }

 private:
  void* pool_ = nullptr;
  pointer_type total_blocks_ = 0;
  pointer_type num_superblocks_ = 0;
  pointer_type num_memory_blocks_ = 0;
  uint32_t superblock_shift_ = 0;
};

template <uint32_t slab_size>
struct device_allocator_context<test_slab_allocator<slab_size>> {
  using host_alloc_type = test_slab_allocator<slab_size>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint32_t bitmap_header_bytes_ =
      host_alloc_type::bitmap_header_bytes_;
  static constexpr uint32_t num_slabs_in_block_ =
      host_alloc_type::num_slabs_in_block_;
  static constexpr uint32_t block_address_bits_ =
      host_alloc_type::block_address_bits_;
  static constexpr uint32_t block_size_ = host_alloc_type::block_size_;
  static constexpr pointer_type invalid_pointer =
      std::numeric_limits<pointer_type>::max();
  static constexpr uint32_t hash_coef_ = 0x5904;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc,
                                            const tile_type& tile)
      : alloc_(alloc), tile_size_(tile_type::size()) {
    static_assert(tile_type::size() == 32 || tile_type::size() == 16);
    initialize(tile);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    using bitmap_type =
        std::conditional_t<tile_type::size() == 32, uint32_t, unsigned long long>;
    pointer_type slab_index = try_allocate_in_current_block<bitmap_type>(tile);
    while (slab_index == invalid_pointer) {
      rehash<bitmap_type>(tile);
      slab_index = try_allocate_in_current_block<bitmap_type>(tile);
    }
    return encode_pointer(superblock_index_, memory_block_index_, slab_index);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_coop(pointer_type p, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      using bitmap_type =
          std::conditional_t<tile_type::size() == 32, uint32_t, unsigned long long>;
      deallocate_in_block<bitmap_type>(p);
    }
  }

  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) noexcept {
    if (tile_size_ == 32) {
      deallocate_in_block<uint32_t>(p);
    }
    else {
      deallocate_in_block<unsigned long long>(p);
    }
    return 0;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(uint32_t sum,
                                                  const tile_type& tile) noexcept {
    (void)sum;
    (void)tile;
  }

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    auto slab_index = p & (num_slabs_in_block_ - 1);
    auto memory_block_index =
        static_cast<pointer_type>((static_cast<uint64_t>(p) >> block_address_bits_) &
                                  alloc_.memory_block_mask_);
    auto superblock_index =
        static_cast<pointer_type>(static_cast<uint64_t>(p) >> alloc_.superblock_shift_);
    auto* block_base = get_block_base(superblock_index, memory_block_index);
    return reinterpret_cast<void*>(block_base + bitmap_header_bytes_ +
                                   static_cast<std::size_t>(slab_index) * slab_size_);
  }

 private:
  const device_instance_type& alloc_;
  pointer_type superblock_index_ = 0;
  pointer_type memory_block_index_ = 0;
  unsigned long long bitmap_cache_ = 0;
  uint32_t tile_size_ = 0;

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type get_tile_id() const {
    return static_cast<pointer_type>(
        (threadIdx.x + blockIdx.x * blockDim.x) / tile_type::size());
  }

  DEVICE_QUALIFIER uint8_t* get_block_base(pointer_type superblock_index,
                                           pointer_type memory_block_index) const {
    auto flat_block_index =
        static_cast<std::size_t>(superblock_index) * alloc_.num_memory_blocks_ +
        memory_block_index;
    return reinterpret_cast<uint8_t*>(alloc_.pool_) + flat_block_index * block_size_;
  }

  template <typename bitmap_type>
  DEVICE_QUALIFIER bitmap_type* current_bitmap_addr(uint32_t lane_rank) const {
    return reinterpret_cast<bitmap_type*>(
               get_block_base(superblock_index_, memory_block_index_)) +
           lane_rank;
  }

  template <typename bitmap_type>
  DEVICE_QUALIFIER void initialize_bitmap_cache(uint32_t lane_rank) {
    bitmap_cache_ = static_cast<unsigned long long>(*current_bitmap_addr<bitmap_type>(lane_rank));
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void initialize(const tile_type& tile) {
    auto tile_id = get_tile_id<tile_type>();
    superblock_index_ = tile_id % alloc_.num_superblocks_;
    memory_block_index_ = first_memory_block_index(tile_id);
    if constexpr (tile_type::size() == 32) {
      initialize_bitmap_cache<uint32_t>(tile.thread_rank());
    }
    else {
      initialize_bitmap_cache<unsigned long long>(tile.thread_rank());
    }
  }

  DEVICE_QUALIFIER pointer_type first_memory_block_index(pointer_type tile_id) const {
    return static_cast<pointer_type>(
        (static_cast<uint64_t>(hash_coef_) * tile_id) % alloc_.num_memory_blocks_);
  }

  template <typename bitmap_type, typename tile_type>
  DEVICE_QUALIFIER pointer_type try_allocate_in_current_block(const tile_type& tile) {
    bitmap_type bitmap = static_cast<bitmap_type>(bitmap_cache_);
    auto* bitmap_addr = current_bitmap_addr<bitmap_type>(tile.thread_rank());
    pointer_type result = invalid_pointer;
    while (result == invalid_pointer) {
      int empty_lane = test_slab_allocator_detail::cuda_ffs(~bitmap) - 1;
      auto free_lane = tile.ballot(empty_lane >= 0);
      if (free_lane == 0) { break; }

      uint32_t src_lane = __ffs(free_lane) - 1;
      if (src_lane == tile.thread_rank()) {
        bitmap_type mask = static_cast<bitmap_type>(1) << empty_lane;
        cuda::atomic_ref<bitmap_type, cuda::thread_scope_device> bitmap_ref(*bitmap_addr);
        bitmap_type old = bitmap_ref.fetch_or(mask, cuda::memory_order_relaxed);
        if ((old & mask) == 0) {
          bitmap = old | mask;
          result = empty_lane + src_lane * sizeof(bitmap_type) * 8;
        }
        else {
          bitmap = old;
        }
      }
      result = tile.shfl(result, src_lane);
    }
    bitmap_cache_ = static_cast<unsigned long long>(bitmap);
    return result;
  }

  template <typename bitmap_type, typename tile_type>
  DEVICE_QUALIFIER void rehash(const tile_type& tile) {
    auto tile_id = get_tile_id<tile_type>();
    auto first_memory_block = first_memory_block_index(tile_id);
    memory_block_index_++;
    if (memory_block_index_ == alloc_.num_memory_blocks_) {
      memory_block_index_ = 0;
    }
    if (memory_block_index_ == first_memory_block) {
      superblock_index_++;
      if (superblock_index_ == alloc_.num_superblocks_) {
        superblock_index_ = 0;
      }
    }
    initialize_bitmap_cache<bitmap_type>(tile.thread_rank());
  }

  DEVICE_QUALIFIER pointer_type encode_pointer(pointer_type superblock_index,
                                               pointer_type memory_block_index,
                                               pointer_type slab_index) const {
    uint64_t encoded =
        (static_cast<uint64_t>(superblock_index) << alloc_.superblock_shift_) |
        (static_cast<uint64_t>(memory_block_index) << block_address_bits_) |
        slab_index;
    return static_cast<pointer_type>(encoded);
  }

  template <typename bitmap_type>
  DEVICE_QUALIFIER void deallocate_in_block(pointer_type p) {
    auto slab_index = p & (num_slabs_in_block_ - 1);
    auto memory_block_index =
        static_cast<pointer_type>((static_cast<uint64_t>(p) >> block_address_bits_) &
                                  alloc_.memory_block_mask_);
    auto superblock_index =
        static_cast<pointer_type>(static_cast<uint64_t>(p) >> alloc_.superblock_shift_);
    auto bitmap_rank = slab_index / (sizeof(bitmap_type) * 8);
    auto bit_index = slab_index % (sizeof(bitmap_type) * 8);
    auto* bitmap_addr =
        reinterpret_cast<bitmap_type*>(get_block_base(superblock_index, memory_block_index)) +
        bitmap_rank;
    bitmap_type mask = static_cast<bitmap_type>(1) << bit_index;
    cuda::atomic_ref<bitmap_type, cuda::thread_scope_device> bitmap_ref(*bitmap_addr);
    [[maybe_unused]] auto old =
        bitmap_ref.fetch_and(~mask, cuda::memory_order_relaxed);
    assert((old & mask) != 0);
  }
};
