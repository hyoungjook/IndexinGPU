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
#include <cstdint>
#include <type_traits>

enum class cuda_memory_order {
  weak,
  relaxed
};

template <typename T, cuda_memory_order order>
struct cuda_memory_32 {};
template <typename T>
struct cuda_memory_32<T, cuda_memory_order::weak> {
  static_assert(sizeof(T) == sizeof(uint32_t));
  using unsigned_type = uint32_t;
  __device__ static inline T load(T* ptr) {
    unsigned_type old;
    asm volatile("ld.weak.global.b32 %0,[%1];"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned_type*>(ptr))
                 : "memory");
    return old;
  }
  __device__ static inline void store(T* ptr, T value) {
    asm volatile("st.weak.global.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                 "r"(*reinterpret_cast<unsigned_type*>(&value))
                 : "memory");
  }
};
template <typename T>
struct cuda_memory_32<T, cuda_memory_order::relaxed> {
  static_assert(sizeof(T) == sizeof(uint32_t));
  using unsigned_type = uint32_t;
  __device__ static inline T load(T* ptr) {
    unsigned_type old;
    asm volatile("ld.relaxed.gpu.b32 %0,[%1];"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned_type*>(ptr))
                 : "memory");
    return old;
  }
  __device__ static inline void store(T* ptr, T value) {
    asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                 "r"(*reinterpret_cast<unsigned_type*>(&value))
                 : "memory");
  }
};

template <typename T, cuda_memory_order order>
struct cuda_memory_64 {};
template <typename T>
struct cuda_memory_64<T, cuda_memory_order::weak> {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using unsigned_type = uint64_t;
  __device__ static inline T load(T* ptr) {
    unsigned_type old;
    asm volatile("ld.weak.global.b64 %0,[%1];"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned_type*>(ptr))
                 : "memory");
    return old;
  }
  __device__ static inline void store(T* ptr, T value) {
    asm volatile("st.weak.global.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                 "r"(*reinterpret_cast<unsigned_type*>(&value))
                 : "memory");
  }
};
template <typename T>
struct cuda_memory_64<T, cuda_memory_order::relaxed> {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using unsigned_type = uint64_t;
  __device__ static inline T load(T* ptr) {
    unsigned_type old;
    asm volatile("ld.relaxed.gpu.b64 %0,[%1];"
                 : "=r"(old)
                 : "l"(reinterpret_cast<unsigned_type*>(ptr))
                 : "memory");
    return old;
  }
  __device__ static inline void store(T* ptr, T value) {
    asm volatile("st.relaxed.gpu.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                 "r"(*reinterpret_cast<unsigned_type*>(&value))
                 : "memory");
  }
};

template <typename T, cuda_memory_order order = cuda_memory_order::weak>
struct cuda_memory
    : public std::conditional<sizeof(T) == 4, cuda_memory_32<T, order>, cuda_memory_64<T, order>>::type {
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

  __device__ static inline void atomic_thread_fence() {
    asm volatile("fence.sc.gpu;" ::: "memory");
  }
};
