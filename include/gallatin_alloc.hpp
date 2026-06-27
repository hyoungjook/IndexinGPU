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

constexpr uint64_t gallatin_segment_bytes = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t gallatin_min_alloc_size = 128ULL;
constexpr uint64_t gallatin_max_alloc_size = 4096ULL;

using global_allocator_type =
    gallatin::allocators::Gallatin<gallatin_segment_bytes,
                                   gallatin_min_alloc_size,
                                   gallatin_max_alloc_size>;

static __device__ global_allocator_type* global_gallatin = nullptr;

__host__ inline bool init_global_allocator(uint64_t num_bytes,
                                           uint64_t seed,
                                           bool print_info = true) {
  auto* local_copy =
      global_allocator_type::generate_on_device(num_bytes, seed, print_info);
  cudaMemcpyToSymbol(global_gallatin, &local_copy,
                     sizeof(global_allocator_type*));
  cudaDeviceSynchronize();
  return local_copy != nullptr;
}

__host__ inline void free_global_allocator() {
  global_allocator_type* local_copy = nullptr;
  cudaMemcpyFromSymbol(&local_copy, global_gallatin,
                       sizeof(global_allocator_type*));
  cudaDeviceSynchronize();
  global_allocator_type::free_on_device(local_copy);
  local_copy = nullptr;
  cudaMemcpyToSymbol(global_gallatin, &local_copy,
                     sizeof(global_allocator_type*));
  cudaDeviceSynchronize();
}

__host__ inline void print_global_stats() {
  global_allocator_type* local_copy = nullptr;
  cudaMemcpyFromSymbol(&local_copy, global_gallatin,
                       sizeof(global_allocator_type*));
  cudaDeviceSynchronize();
  local_copy->print_info();
}

__device__ inline void* global_malloc(uint64_t num_bytes) {
  return global_gallatin->malloc(num_bytes);
}

__device__ inline void global_free(void* ptr) {
  global_gallatin->free(ptr);
}

#ifdef GALLATIN_STATIC_COUNTER
// Stateful (per-thread context) fast path: ONE atomic on the cached slot.
__device__ inline void* global_gstatic_fast(int cidx, uint64_t cbase,
                                            unsigned int cgen, uint64_t alloc_size) {
  return global_gallatin->gstatic_fast(cidx, cbase, cgen, alloc_size);
}
__device__ inline void* global_gstatic_fast_grouped(int& cidx, uint64_t& cbase,
                                                    unsigned int& cgen,
                                                    uint16_t tree_id,
                                                    uint64_t alloc_size) {
  return global_gallatin->gstatic_fast_grouped(cidx, cbase, cgen, tree_id,
                                               alloc_size);
}
__device__ inline void* global_gstatic_slow(uint16_t tree_id, uint64_t alloc_size,
                                            int& ci, uint64_t& cb, unsigned int& cg) {
  return global_gallatin->gstatic_slow(tree_id, alloc_size, ci, cb, cg);
}
__device__ inline void* global_gstatic_prefetch(
    int& cidx, uint64_t& cbase, unsigned int& cgen, unsigned long long& pf_merged,
    int& pf_cidx, uint64_t& pf_cbase, unsigned int& pf_cgen, bool& pf_valid,
    uint16_t tree_id, uint64_t alloc_size) {
  return global_gallatin->gstatic_prefetch(cidx, cbase, cgen, pf_merged, pf_cidx,
                                           pf_cbase, pf_cgen, pf_valid, tree_id,
                                           alloc_size);
}
__device__ inline void* global_gstatic_prefetch_drain(unsigned long long pf_merged,
                                                      uint64_t pf_cbase,
                                                      unsigned int pf_cgen,
                                                      uint64_t alloc_size) {
  return global_gallatin->gstatic_prefetch_drain(pf_merged, pf_cbase, pf_cgen,
                                                 alloc_size);
}
__device__ inline uint16_t global_tree_id(uint64_t num_bytes) {
  return global_gallatin->get_tree_id_from_size(num_bytes);
}
__device__ inline uint64_t global_tree_alloc_size(uint16_t tree_id) {
  return global_gallatin->table->get_tree_alloc_size(tree_id);
}
#endif

__device__ inline uint64_t global_memory_base() {
  return reinterpret_cast<uint64_t>(global_gallatin->table->memory);
}

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
  static constexpr uint64_t gallatin_segment_bytes_ =
      gallatin_alloc_detail::gallatin_segment_bytes;
  static constexpr uint64_t gallatin_min_alloc_size_ =
      gallatin_alloc_detail::gallatin_min_alloc_size;
  static constexpr uint64_t gallatin_max_alloc_size_ =
      gallatin_alloc_detail::gallatin_max_alloc_size;
  using gallatin_allocator_type = gallatin_alloc_detail::global_allocator_type;

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

    initialized_ = gallatin_alloc_detail::init_global_allocator(max_bytes, seed,
                                                                print_info);
    if (!initialized_) {
      std::cerr << "gallatin_allocator: Gallatin initialization failed"
                << std::endl;
      std::abort();
    }
  }

  ~gallatin_allocator() {
    if (initialized_) {
      gallatin_alloc_detail::free_global_allocator();
    }
  }

  gallatin_allocator(const gallatin_allocator& other) = delete;
  gallatin_allocator& operator=(const gallatin_allocator& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{total_slots_};
  }

  void print_stats() const {
    std::cout << "gallatin_allocator(" << slab_size_ << "B slabs, "
              << effective_slab_size_ << "B Gallatin slices): "
              << total_slots_ << " max slots" << std::endl;
    gallatin_alloc_detail::print_global_stats();
  }

private:
  bool initialized_ = false;
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

    initialized_ = gallatin_alloc_detail::init_global_allocator(
        gallatin_bytes, seed, print_info);
    if (!initialized_) {
      std::cerr << "gallatin_linear_allocator: Gallatin initialization failed"
                << std::endl;
      std::abort();
    }

    cuda_try(cudaMalloc(&linear_count_, sizeof(size_type)));
    cuda_try(cudaMemset(linear_count_, 0x00, sizeof(size_type)));
  }

  ~gallatin_linear_allocator() {
    if (initialized_) {
      gallatin_alloc_detail::free_global_allocator();
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
    return device_instance_type{total_slots_, linear_pool_, linear_count_,
                                linear_capacity_};
  }

  void print_stats() const {
    size_type h_linear_count = 0;
    cuda_try(cudaMemcpy(&h_linear_count, linear_count_, sizeof(size_type),
                        cudaMemcpyDeviceToHost));
    std::cout << "gallatin_linear_allocator(" << slab_size_ << "B slabs, "
              << effective_slab_size_ << "B Gallatin slices): "
              << total_slots_ << " max slots, " << h_linear_count << "/"
              << linear_capacity_ << " linear entries" << std::endl;
    gallatin_alloc_detail::print_global_stats();
  }

private:
  bool initialized_ = false;
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
    // Cache the heap base ONCE per kernel. pointer_to_handle()/address() would
    // otherwise chase global_gallatin->table->memory (3 dependent global loads)
    // on every alloc/free -- a pure long_scoreboard cost. The base is constant
    // for the kernel, so resolve it here and keep it register-resident.
    mbase_ = gallatin_alloc_detail::global_memory_base();
#ifdef GALLATIN_STATIC_COUNTER
    // Per-(tile-leader) resident context: cache the slot/base/gen across the
    // allocation loop so the hot path is one atomic + a register compare.
    cidx_ = -1;
    cbase_ = 0;
    cgen_ = 0;
    tree_id_ = gallatin_alloc_detail::global_tree_id(slab_size_);
    talloc_ = gallatin_alloc_detail::global_tree_alloc_size(tree_id_);
    // The tree's slice size is provably the compile-time effective_slab_size_
    // (smallest pow2 bucket >= slab_size). Verified here so the fast path can
    // use the constant (folds the count*size multiply) instead of talloc_.
    assert(talloc_ == effective_slab_size_);
#ifdef GALLATIN_PREFETCH
    pf_valid_ = false;  // no reservation outstanding yet
#endif
#endif
  }

#ifdef GALLATIN_PREFETCH
  // Return the one outstanding (issued-but-unconsumed) reservation so no
  // allocation is ever lost. pf_valid_ is only ever set on the tile leader, so
  // this is implicitly leader-only. Freeing the reserved-but-unwritten slice is
  // identical to an app alloc+free -> the block's recycle accounting stays exact.
  DEVICE_QUALIFIER ~device_allocator_context() {
    if (pf_valid_) {
      void* leftover = gallatin_alloc_detail::global_gstatic_prefetch_drain(
          pf_merged_, pf_cbase_, pf_cgen_, effective_slab_size_);
      if (leftover != nullptr) gallatin_alloc_detail::global_free(leftover);
    }
  }
#endif

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    uint64_t raw_addr = 0;
    if (tile.thread_rank() == 0) {
#ifdef GALLATIN_STATIC_COUNTER
      // Try the cached slot (1 atomic, no load); on miss re-resolve safely and
      // refresh the cached {cidx_, cbase_, cgen_}.
#if defined(GALLATIN_PREFETCH)
      // Software-pipelined: consume the reservation issued LAST call (its atomic
      // has retired -> no stall) and issue the next undecoded, so its latency
      // overlaps the caller's insert work. Runs fast+slow internally and always
      // returns a valid slice (or genuine-exhaustion null). The one outstanding
      // reservation is returned at teardown (~device_allocator_context).
      void* raw_ptr = gallatin_alloc_detail::global_gstatic_prefetch(
          cidx_, cbase_, cgen_, pf_merged_, pf_cidx_, pf_cbase_, pf_cgen_,
          pf_valid_, tree_id_, effective_slab_size_);
#elif defined(GALLATIN_GROUPED)
      // Warp-coalesced reservation: the active tile leaders that share a warp
      // (and tree) claim a contiguous run with ONE atomicAdd -- Gallatin's
      // native coalescing, applied to the static counter. Only tile sizes < 32
      // can have >1 leader per warp; tile32 (1 leader/warp) skips the coalescing
      // machinery entirely (it would be pure overhead with nothing to coalesce).
      void* raw_ptr;
      if constexpr (tile_type::size() < 32) {
        raw_ptr = gallatin_alloc_detail::global_gstatic_fast_grouped(
            cidx_, cbase_, cgen_, tree_id_, effective_slab_size_);
      } else {
        raw_ptr = gallatin_alloc_detail::global_gstatic_fast(
            cidx_, cbase_, cgen_, effective_slab_size_);
      }
      if (raw_ptr == nullptr) {
        raw_ptr = gallatin_alloc_detail::global_gstatic_slow(
            tree_id_, talloc_, cidx_, cbase_, cgen_);
      }
#else
      void* raw_ptr = gallatin_alloc_detail::global_gstatic_fast(
          cidx_, cbase_, cgen_, effective_slab_size_);
      if (raw_ptr == nullptr) {
        raw_ptr = gallatin_alloc_detail::global_gstatic_slow(
            tree_id_, talloc_, cidx_, cbase_, cgen_);
      }
#endif
#else
      auto* raw_ptr = gallatin_alloc_detail::global_malloc(slab_size_);
#endif
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
      gallatin_alloc_detail::global_free(address(p));
    }
  }

  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) {
    gallatin_alloc_detail::global_free(address(p));
    return 0;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(uint32_t sum,
                                                  const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    assert(p < alloc_.total_slots_);
    auto raw_addr = mbase_ + static_cast<uint64_t>(p) * effective_slab_size_;
    return reinterpret_cast<void*>(raw_addr);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint64_t effective_slab_size_ =
      host_alloc_type::effective_slab_size_;

  const device_instance_type& alloc_;
  uint64_t mbase_ = 0;     // cached heap base (global_gallatin->table->memory)

#ifdef GALLATIN_STATIC_COUNTER
  int cidx_ = -1;          // cached static-counter slot index (-1 = none)
  uint64_t cbase_ = 0;     // cached slot block base address
  unsigned int cgen_ = 0;  // cached slot generation (validates cbase against swaps)
  uint16_t tree_id_ = 0;   // tree for slab_size_ (resolved once)
  uint64_t talloc_ = 0;    // tree slice size (slice stride within a block)
#ifdef GALLATIN_PREFETCH
  // One outstanding pipelined reservation (leader-only): the raw atomic result
  // (undecoded -> its latency is hidden) plus the slot it was issued against.
  unsigned long long pf_merged_ = 0;
  int pf_cidx_ = -1;
  uint64_t pf_cbase_ = 0;
  unsigned int pf_cgen_ = 0;
  bool pf_valid_ = false;
#endif
#endif

  DEVICE_QUALIFIER pointer_type pointer_to_handle(uint64_t raw_addr) const {
    auto base_addr = mbase_;
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
      auto* raw_ptr = gallatin_alloc_detail::global_malloc(slab_size_);
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
      gallatin_alloc_detail::global_free(address(p));
    }
  }

  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) {
    gallatin_alloc_detail::global_free(address(p));
    return 0;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(
      uint32_t sum,
      const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    assert(p < alloc_.total_slots_);
    auto raw_addr = gallatin_alloc_detail::global_memory_base() +
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

#ifdef GALLATIN_STATIC_COUNTER
  int cidx_ = -1;          // cached static-counter slot index (-1 = none)
  uint64_t cbase_ = 0;     // cached slot block base address
  unsigned int cgen_ = 0;  // cached slot generation (validates cbase against swaps)
  uint16_t tree_id_ = 0;   // tree for slab_size_ (resolved once)
  uint64_t talloc_ = 0;    // tree slice size (slice stride within a block)
#endif

  DEVICE_QUALIFIER pointer_type pointer_to_handle(uint64_t raw_addr) const {
    auto base_addr = gallatin_alloc_detail::global_memory_base();
    assert(raw_addr >= base_addr);
    auto offset = raw_addr - base_addr;
    assert((offset % effective_slab_size_) == 0);
    auto handle = offset / effective_slab_size_;
    assert(handle < alloc_.total_slots_);
    return static_cast<pointer_type>(handle);
  }
};
