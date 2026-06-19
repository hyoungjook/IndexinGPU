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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <device_context.hpp>
#include <gallatin/allocators/gallatin.cuh>
#include <macros.hpp>
#include <utils.hpp>

namespace gallatin_alloc_detail {

constexpr uint64_t max_u64(uint64_t a, uint64_t b) {
  return a > b ? a : b;
}

constexpr uint64_t bit_ceil(uint64_t value) {
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value |= value >> 32;
  return value + 1;
}

constexpr uint32_t log2_power_of_two(uint64_t value) {
  uint32_t result = 0;
  while ((1ULL << result) < value) {
    result++;
  }
  return result;
}

}  // namespace gallatin_alloc_detail

template <uint32_t slab_size = 128>
struct gallatin_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;

  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint64_t gallatin_segment_bytes_ = 16ULL * 1024ULL * 1024ULL;
  static constexpr uint64_t gallatin_min_alloc_size_ = 16ULL;
  static constexpr uint64_t gallatin_max_alloc_size_ = 4096ULL;
  using gallatin_allocator_type =
      gallatin::allocators::Gallatin<gallatin_segment_bytes_,
                                     gallatin_min_alloc_size_,
                                     gallatin_max_alloc_size_>;

  static constexpr uint64_t effective_slab_size_ =
      gallatin_alloc_detail::bit_ceil(
          gallatin_alloc_detail::max_u64(slab_size_,
                                         gallatin_min_alloc_size_));
  static constexpr uint32_t gallatin_num_trees_ =
      gallatin_alloc_detail::log2_power_of_two(gallatin_max_alloc_size_) -
      gallatin_alloc_detail::log2_power_of_two(gallatin_min_alloc_size_) + 1;

  static_assert(slab_size_ > 0);
  static_assert(slab_size_ <= gallatin_max_alloc_size_);
  static_assert(effective_slab_size_ >= gallatin_min_alloc_size_);
  static_assert(effective_slab_size_ <= gallatin_max_alloc_size_);

  struct device_instance_type {
    gallatin_allocator_type* allocator_;
    pointer_type total_slots_;
  };

  gallatin_allocator(float pool_ratio = 0.9f,
                     uint64_t seed = 42,
                     bool print_info = false) {
    auto meminfo = utils::compute_device_memory_usage();
    auto max_bytes =
        static_cast<uint64_t>(static_cast<double>(meminfo.total_bytes) *
                              static_cast<double>(pool_ratio));
    auto num_segments = max_bytes / gallatin_segment_bytes_;
    if (num_segments < gallatin_num_trees_) {
      std::cerr << "gallatin_allocator: pool has " << num_segments
                << " " << gallatin_segment_bytes_
                << "B segments, but Gallatin requires at least "
                << gallatin_num_trees_ << " segments" << std::endl;
      std::abort();
    }

    auto capacity_bytes = num_segments * gallatin_segment_bytes_;
    auto total_slots = capacity_bytes / effective_slab_size_;
    if (total_slots > std::numeric_limits<pointer_type>::max()) {
      std::cerr << "gallatin_allocator: compact pointer exceeds uint32 limit"
                << std::endl;
      std::abort();
    }
    total_slots_ = static_cast<pointer_type>(total_slots);

    allocator_ = gallatin_allocator_type::generate_on_device(max_bytes, seed,
                                                             print_info);
    if (allocator_ == nullptr) {
      std::cerr << "gallatin_allocator: Gallatin initialization failed"
                << std::endl;
      std::abort();
    }
  }

  ~gallatin_allocator() {
    if (allocator_ != nullptr) {
      gallatin_allocator_type::free_on_device(allocator_);
    }
  }

  gallatin_allocator(const gallatin_allocator& other) = delete;
  gallatin_allocator& operator=(const gallatin_allocator& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{allocator_, total_slots_};
  }

  void print_stats() const {
    std::cout << "gallatin_allocator(" << slab_size_ << "B slabs, "
              << effective_slab_size_ << "B Gallatin slices): "
              << total_slots_ << " max slots" << std::endl;
    allocator_->print_info();
  }

private:
  gallatin_allocator_type* allocator_ = nullptr;
  pointer_type total_slots_ = 0;
};

template <uint32_t slab_size = 128>
struct gallatin_linear_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;

  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint64_t gallatin_segment_bytes_ =
      gallatin_allocator<slab_size>::gallatin_segment_bytes_;
  static constexpr uint64_t gallatin_min_alloc_size_ =
      gallatin_allocator<slab_size>::gallatin_min_alloc_size_;
  static constexpr uint64_t gallatin_max_alloc_size_ =
      gallatin_allocator<slab_size>::gallatin_max_alloc_size_;
  static constexpr uint64_t effective_slab_size_ =
      gallatin_allocator<slab_size>::effective_slab_size_;
  static constexpr uint32_t gallatin_num_trees_ =
      gallatin_allocator<slab_size>::gallatin_num_trees_;
  using gallatin_allocator_type =
      typename gallatin_allocator<slab_size>::gallatin_allocator_type;

  static_assert(slab_size_ > 0);
  static_assert(slab_size_ <= gallatin_max_alloc_size_);

  struct device_instance_type {
    gallatin_allocator_type* allocator_;
    pointer_type total_slots_;
    size_type* linear_pool_;
    size_type* linear_count_;
    size_type linear_capacity_;
  };

  gallatin_linear_allocator(float pool_ratio = 0.9f,
                            float initial_slab_ratio = 0.7f,
                            uint64_t seed = 42,
                            bool print_info = false) {
    if (!(pool_ratio > 0.0f)) {
      std::cerr << "gallatin_linear_allocator: pool_ratio must be positive"
                << std::endl;
      std::abort();
    }
    if (!(initial_slab_ratio >= 0.0f && initial_slab_ratio < 1.0f)) {
      std::cerr << "gallatin_linear_allocator: initial_slab_ratio must be in "
                   "[0, 1)"
                << std::endl;
      std::abort();
    }

    auto meminfo = utils::compute_device_memory_usage();
    auto max_bytes =
        static_cast<uint64_t>(static_cast<double>(meminfo.total_bytes) *
                              static_cast<double>(pool_ratio));
    auto linear_bytes = static_cast<uint64_t>(
        static_cast<double>(max_bytes) *
        (1.0 - static_cast<double>(initial_slab_ratio)));
    auto gallatin_bytes = max_bytes - linear_bytes;
    auto num_segments = gallatin_bytes / gallatin_segment_bytes_;
    if (num_segments < gallatin_num_trees_) {
      std::cerr << "gallatin_linear_allocator: pool has " << num_segments
                << " " << gallatin_segment_bytes_
                << "B segments, but Gallatin requires at least "
                << gallatin_num_trees_ << " segments" << std::endl;
      std::abort();
    }

    auto capacity_bytes = num_segments * gallatin_segment_bytes_;
    auto total_slots = capacity_bytes / effective_slab_size_;
    if (total_slots > std::numeric_limits<pointer_type>::max()) {
      std::cerr
          << "gallatin_linear_allocator: compact pointer exceeds uint32 limit"
          << std::endl;
      std::abort();
    }
    total_slots_ = static_cast<pointer_type>(total_slots);

    auto linear_entries = linear_bytes / sizeof(size_type);
    if (linear_entries > std::numeric_limits<size_type>::max()) {
      linear_entries = std::numeric_limits<size_type>::max();
    }
    linear_capacity_ = static_cast<size_type>(linear_entries);
    linear_capacity_ = linear_capacity_ / 32 * 32;
    if (linear_capacity_ == 0) {
      std::cerr << "gallatin_linear_allocator: linear capacity is zero"
                << std::endl;
      std::abort();
    }

    cuda_try(cudaMalloc(&linear_pool_,
                        static_cast<std::size_t>(linear_capacity_) *
                            sizeof(size_type)));

    allocator_ = gallatin_allocator_type::generate_on_device(gallatin_bytes, seed,
                                                             print_info);
    if (allocator_ == nullptr) {
      std::cerr << "gallatin_linear_allocator: Gallatin initialization failed"
                << std::endl;
      std::abort();
    }

    cuda_try(cudaMalloc(&linear_count_, sizeof(size_type)));
    cuda_try(cudaMemset(linear_count_, 0x00, sizeof(size_type)));
  }

  ~gallatin_linear_allocator() {
    if (allocator_ != nullptr) {
      gallatin_allocator_type::free_on_device(allocator_);
    }
    if (linear_pool_ != nullptr) {
      cuda_try(cudaFree(linear_pool_));
    }
    if (linear_count_ != nullptr) {
      cuda_try(cudaFree(linear_count_));
    }
  }

  gallatin_linear_allocator(const gallatin_linear_allocator& other) = delete;
  gallatin_linear_allocator& operator=(const gallatin_linear_allocator& other) =
      delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{allocator_, total_slots_, linear_pool_,
                                linear_count_, linear_capacity_};
  }

  void print_stats() const {
    size_type h_linear_count = 0;
    cuda_try(cudaMemcpy(&h_linear_count, linear_count_, sizeof(size_type),
                        cudaMemcpyDeviceToHost));
    std::cout << "gallatin_linear_allocator(" << slab_size_ << "B slabs, "
              << effective_slab_size_ << "B Gallatin slices): "
              << total_slots_ << " max slots, " << h_linear_count << "/"
              << linear_capacity_ << " linear entries" << std::endl;
    allocator_->print_info();
  }

private:
  gallatin_allocator_type* allocator_ = nullptr;
  pointer_type total_slots_ = 0;
  size_type* linear_pool_ = nullptr;
  size_type* linear_count_ = nullptr;
  size_type linear_capacity_ = 0;
};

template <uint32_t slab_size>
struct device_allocator_context<gallatin_allocator<slab_size>> {
  using host_alloc_type = gallatin_allocator<slab_size>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc,
                                            const tile_type& tile)
      : alloc_(alloc) {
    static_assert(tile_type::size() == 32 || tile_type::size() == 16);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    uint64_t raw_addr = 0;
    if (tile.thread_rank() == 0) {
      auto* raw_ptr = alloc_.allocator_->malloc(slab_size_);
      if (raw_ptr == nullptr) {
        printf("gallatin_allocator: allocation failed for %u bytes\n",
               slab_size_);
        asm volatile("trap;");
      }
      raw_addr = reinterpret_cast<uint64_t>(raw_ptr);
    }
    raw_addr = tile.shfl(raw_addr, 0);
    return pointer_to_handle(raw_addr);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_coop(pointer_type p, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      alloc_.allocator_->free(address(p));
    }
  }

  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) {
    alloc_.allocator_->free(address(p));
    return 0;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(uint32_t sum,
                                                  const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    assert(p < alloc_.total_slots_);
    auto raw_addr = reinterpret_cast<uint64_t>(alloc_.allocator_->table->memory) +
                    static_cast<uint64_t>(p) * effective_slab_size_;
    return reinterpret_cast<void*>(raw_addr);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint64_t effective_slab_size_ =
      host_alloc_type::effective_slab_size_;

  const device_instance_type& alloc_;

  DEVICE_QUALIFIER pointer_type pointer_to_handle(uint64_t raw_addr) const {
    auto base_addr = reinterpret_cast<uint64_t>(alloc_.allocator_->table->memory);
    assert(raw_addr >= base_addr);
    auto offset = raw_addr - base_addr;
    assert((offset % effective_slab_size_) == 0);
    auto handle = offset / effective_slab_size_;
    assert(handle < alloc_.total_slots_);
    return static_cast<pointer_type>(handle);
  }
};

template <uint32_t slab_size>
struct device_allocator_context<gallatin_linear_allocator<slab_size>> {
  using host_alloc_type = gallatin_linear_allocator<slab_size>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc,
                                            const tile_type& tile)
      : alloc_(alloc) {
    static_assert(tile_type::size() == 32 || tile_type::size() == 16);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    uint64_t raw_addr = 0;
    if (tile.thread_rank() == 0) {
      auto* raw_ptr = alloc_.allocator_->malloc(slab_size_);
      if (raw_ptr == nullptr) {
        printf("gallatin_linear_allocator: allocation failed for %u bytes\n",
               slab_size_);
        asm volatile("trap;");
      }
      raw_addr = reinterpret_cast<uint64_t>(raw_ptr);
    }
    raw_addr = tile.shfl(raw_addr, 0);
    return pointer_to_handle(raw_addr);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_coop(pointer_type p, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      alloc_.allocator_->free(address(p));
    }
  }

  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) {
    alloc_.allocator_->free(address(p));
    return 0;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(
      uint32_t sum,
      const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    assert(p < alloc_.total_slots_);
    auto raw_addr = reinterpret_cast<uint64_t>(alloc_.allocator_->table->memory) +
                    static_cast<uint64_t>(p) * effective_slab_size_;
    return reinterpret_cast<void*>(raw_addr);
  }

  DEVICE_QUALIFIER void* get_linear() const {
    return reinterpret_cast<void*>(alloc_.linear_pool_ +
                                   alloc_.linear_capacity_);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER size_type reallocate_linear(size_type size,
                                               const tile_type& tile) {
    size_type result = size;
    if (tile.thread_rank() == 0) {
      auto requested = min(size, alloc_.linear_capacity_);
      cuda::atomic_ref<size_type, cuda::thread_scope_device> linear_count_ref(
          *alloc_.linear_count_);
      auto current = linear_count_ref.load(cuda::memory_order_relaxed);
      while (current < requested) {
        if (linear_count_ref.compare_exchange_strong(
                current, requested, cuda::memory_order_acquire,
                cuda::memory_order_relaxed)) {
          current = requested;
          break;
        }
      }
      result = current < requested ? current : requested;
    }
    return tile.shfl(result, 0);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint64_t effective_slab_size_ =
      host_alloc_type::effective_slab_size_;

  const device_instance_type& alloc_;

  DEVICE_QUALIFIER pointer_type pointer_to_handle(uint64_t raw_addr) const {
    auto base_addr = reinterpret_cast<uint64_t>(alloc_.allocator_->table->memory);
    assert(raw_addr >= base_addr);
    auto offset = raw_addr - base_addr;
    assert((offset % effective_slab_size_) == 0);
    auto handle = offset / effective_slab_size_;
    assert(handle < alloc_.total_slots_);
    return static_cast<pointer_type>(handle);
  }
};
