#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <gpu_dycuckoo_backend.hpp>
#include <macros.hpp>

template <std::uint32_t Slices>
struct DycuckooLockKey {
  std::uint32_t data[Slices];

  __host__ __device__ constexpr explicit operator std::uint32_t() const {
    return data[Slices - 1];
  }

  __host__ __device__ constexpr std::uint32_t operator&(std::uint32_t mask) const {
    return static_cast<std::uint32_t>(*this) & mask;
  }

  __host__ __device__ constexpr bool operator==(const DycuckooLockKey& other) const {
    for (std::uint32_t i = 0; i < Slices; i++) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  __host__ __device__ constexpr bool operator!=(const DycuckooLockKey& other) const {
    return !(*this == other);
  }
};

#define DataLayout DycuckooLockDataLayoutBase
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#include "../baselines/DyCuckoo/dynamicHash_lock/data/data_layout.cuh"
#undef hashers
#undef cuckoo_helpers
#undef DataLayout

template <
  typename Key = std::uint32_t,
  std::uint32_t KeyBits = 32,
  std::uint32_t ValBits = 32,
  std::uint32_t EmptyKey = 0,
  std::uint32_t EmptyValue = 0,
  std::uint32_t BucketSize = 16,
  std::uint32_t TableNum = 4,
  std::uint32_t errorTableLen = 10000,
  std::uint32_t lockTag = 1,
  std::uint32_t unlockTag = 0>
using DycuckooLockDataLayoutKey1 =
  DycuckooLockDataLayoutBase<Key, KeyBits, ValBits, EmptyKey, EmptyValue, BucketSize, TableNum, errorTableLen, lockTag, unlockTag>;

template <
  typename Key = DycuckooLockKey<2>,
  std::uint32_t KeyBits = 64,
  std::uint32_t ValBits = 32,
  std::uint32_t EmptyKey = 0,
  std::uint32_t EmptyValue = 0,
  std::uint32_t BucketSize = 16,
  std::uint32_t TableNum = 4,
  std::uint32_t errorTableLen = 10000,
  std::uint32_t lockTag = 1,
  std::uint32_t unlockTag = 0>
using DycuckooLockDataLayoutKey2 =
  DycuckooLockDataLayoutBase<Key, KeyBits, ValBits, EmptyKey, EmptyValue, BucketSize, TableNum, errorTableLen, lockTag, unlockTag>;

template <
  typename Key = DycuckooLockKey<4>,
  std::uint32_t KeyBits = 128,
  std::uint32_t ValBits = 32,
  std::uint32_t EmptyKey = 0,
  std::uint32_t EmptyValue = 0,
  std::uint32_t BucketSize = 16,
  std::uint32_t TableNum = 4,
  std::uint32_t errorTableLen = 10000,
  std::uint32_t lockTag = 1,
  std::uint32_t unlockTag = 0>
using DycuckooLockDataLayoutKey4 =
  DycuckooLockDataLayoutBase<Key, KeyBits, ValBits, EmptyKey, EmptyValue, BucketSize, TableNum, errorTableLen, lockTag, unlockTag>;

template <
  typename Key = DycuckooLockKey<8>,
  std::uint32_t KeyBits = 256,
  std::uint32_t ValBits = 32,
  std::uint32_t EmptyKey = 0,
  std::uint32_t EmptyValue = 0,
  std::uint32_t BucketSize = 16,
  std::uint32_t TableNum = 4,
  std::uint32_t errorTableLen = 10000,
  std::uint32_t lockTag = 1,
  std::uint32_t unlockTag = 0>
using DycuckooLockDataLayoutKey8 =
  DycuckooLockDataLayoutBase<Key, KeyBits, ValBits, EmptyKey, EmptyValue, BucketSize, TableNum, errorTableLen, lockTag, unlockTag>;

template <
  typename Key = DycuckooLockKey<16>,
  std::uint32_t KeyBits = 512,
  std::uint32_t ValBits = 32,
  std::uint32_t EmptyKey = 0,
  std::uint32_t EmptyValue = 0,
  std::uint32_t BucketSize = 16,
  std::uint32_t TableNum = 4,
  std::uint32_t errorTableLen = 10000,
  std::uint32_t lockTag = 1,
  std::uint32_t unlockTag = 0>
using DycuckooLockDataLayoutKey16 =
  DycuckooLockDataLayoutBase<Key, KeyBits, ValBits, EmptyKey, EmptyValue, BucketSize, TableNum, errorTableLen, lockTag, unlockTag>;

template <typename...>
using DycuckooLockDataLayoutValue2 =
  DycuckooLockDataLayoutBase<std::uint32_t, 32, 64, 0, 0, 16, 4, 10000, 1, 0>;

template <typename...>
using DycuckooLockDataLayoutValue4 =
  DycuckooLockDataLayoutBase<std::uint32_t, 32, 128, 0, 0, 16, 4, 10000, 1, 0>;

template <typename...>
using DycuckooLockDataLayoutValue8 =
  DycuckooLockDataLayoutBase<std::uint32_t, 32, 256, 0, 0, 16, 4, 10000, 1, 0>;

template <typename...>
using DycuckooLockDataLayoutValue16 =
  DycuckooLockDataLayoutBase<std::uint32_t, 32, 512, 0, 0, 16, 4, 10000, 1, 0>;

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutKey1
#define DynamicHash DycuckooLockDynamicHashKey1
#define DynamicCuckoo DycuckooLockDynamicCuckooKey1
#define cuckoo_table DycuckooLockCuckooTableKey1
#define error_table DycuckooLockErrorTableKey1
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_key1
#define cg dycuckoo_lock_cg_key1
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutKey2
#define DynamicHash DycuckooLockDynamicHashKey2
#define DynamicCuckoo DycuckooLockDynamicCuckooKey2
#define cuckoo_table DycuckooLockCuckooTableKey2
#define error_table DycuckooLockErrorTableKey2
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_key2
#define cg dycuckoo_lock_cg_key2
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutKey4
#define DynamicHash DycuckooLockDynamicHashKey4
#define DynamicCuckoo DycuckooLockDynamicCuckooKey4
#define cuckoo_table DycuckooLockCuckooTableKey4
#define error_table DycuckooLockErrorTableKey4
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_key4
#define cg dycuckoo_lock_cg_key4
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutKey8
#define DynamicHash DycuckooLockDynamicHashKey8
#define DynamicCuckoo DycuckooLockDynamicCuckooKey8
#define cuckoo_table DycuckooLockCuckooTableKey8
#define error_table DycuckooLockErrorTableKey8
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_key8
#define cg dycuckoo_lock_cg_key8
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutKey16
#define DynamicHash DycuckooLockDynamicHashKey16
#define DynamicCuckoo DycuckooLockDynamicCuckooKey16
#define cuckoo_table DycuckooLockCuckooTableKey16
#define error_table DycuckooLockErrorTableKey16
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_key16
#define cg dycuckoo_lock_cg_key16
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutValue2
#define DynamicHash DycuckooLockDynamicHashValue2
#define DynamicCuckoo DycuckooLockDynamicCuckooValue2
#define cuckoo_table DycuckooLockCuckooTableValue2
#define error_table DycuckooLockErrorTableValue2
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_value2
#define cg dycuckoo_lock_cg_value2
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutValue4
#define DynamicHash DycuckooLockDynamicHashValue4
#define DynamicCuckoo DycuckooLockDynamicCuckooValue4
#define cuckoo_table DycuckooLockCuckooTableValue4
#define error_table DycuckooLockErrorTableValue4
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_value4
#define cg dycuckoo_lock_cg_value4
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutValue8
#define DynamicHash DycuckooLockDynamicHashValue8
#define DynamicCuckoo DycuckooLockDynamicCuckooValue8
#define cuckoo_table DycuckooLockCuckooTableValue8
#define error_table DycuckooLockErrorTableValue8
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_value8
#define cg dycuckoo_lock_cg_value8
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

#undef DYNAMIC_HASH_H
#undef DYNAMIC_CUCKOO_H
#define DataLayout DycuckooLockDataLayoutValue16
#define DynamicHash DycuckooLockDynamicHashValue16
#define DynamicCuckoo DycuckooLockDynamicCuckooValue16
#define cuckoo_table DycuckooLockCuckooTableValue16
#define error_table DycuckooLockErrorTableValue16
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#define ch dycuckoo_lock_ch_value16
#define cg dycuckoo_lock_cg_value16
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef cg
#undef ch
#undef hashers
#undef cuckoo_helpers
#undef error_table
#undef cuckoo_table
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

namespace {

struct lock_handle {
  std::uint32_t keylen_max;
  std::uint32_t valuelen_max;
  void* index;
};

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
struct lock_traits;

template <>
struct lock_traits<1, 1> {
  using layout_type = DycuckooLockDataLayoutKey1<>;
  using index_type = DycuckooLockDynamicCuckooKey1<512, 512>;
};

template <>
struct lock_traits<2, 1> {
  using layout_type = DycuckooLockDataLayoutKey2<>;
  using index_type = DycuckooLockDynamicCuckooKey2<512, 512>;
};

template <>
struct lock_traits<4, 1> {
  using layout_type = DycuckooLockDataLayoutKey4<>;
  using index_type = DycuckooLockDynamicCuckooKey4<512, 512>;
};

template <>
struct lock_traits<8, 1> {
  using layout_type = DycuckooLockDataLayoutKey8<>;
  using index_type = DycuckooLockDynamicCuckooKey8<512, 512>;
};

template <>
struct lock_traits<16, 1> {
  using layout_type = DycuckooLockDataLayoutKey16<>;
  using index_type = DycuckooLockDynamicCuckooKey16<512, 512>;
};

template <>
struct lock_traits<1, 2> {
  using layout_type = DycuckooLockDataLayoutValue2<>;
  using index_type = DycuckooLockDynamicCuckooValue2<512, 512>;
};

template <>
struct lock_traits<1, 4> {
  using layout_type = DycuckooLockDataLayoutValue4<>;
  using index_type = DycuckooLockDynamicCuckooValue4<512, 512>;
};

template <>
struct lock_traits<1, 8> {
  using layout_type = DycuckooLockDataLayoutValue8<>;
  using index_type = DycuckooLockDynamicCuckooValue8<512, 512>;
};

template <>
struct lock_traits<1, 16> {
  using layout_type = DycuckooLockDataLayoutValue16<>;
  using index_type = DycuckooLockDynamicCuckooValue16<512, 512>;
};

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void validate_lock_layout() {
  using key_type = typename lock_traits<KeyLen, ValueLen>::layout_type::key_t;
  using value_type = typename lock_traits<KeyLen, ValueLen>::layout_type::value_t;
  static_assert(sizeof(key_type) == KeyLen * sizeof(std::uint32_t));
  static_assert(alignof(key_type) == alignof(std::uint32_t));
  static_assert(sizeof(value_type) == ValueLen * sizeof(std::uint32_t));
  static_assert(alignof(value_type) == alignof(std::uint32_t));
}

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void* create_typed_index(std::uint32_t init_kv_num,
                         int small_batch_size,
                         double lower_bound,
                         double upper_bound) {
  validate_lock_layout<KeyLen, ValueLen>();
  using index_type = typename lock_traits<KeyLen, ValueLen>::index_type;
  return new index_type(init_kv_num, small_batch_size, lower_bound, upper_bound);
}

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void destroy_typed_index(void* index) {
  using index_type = typename lock_traits<KeyLen, ValueLen>::index_type;
  delete reinterpret_cast<index_type*>(index);
}

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void insert_typed(lock_handle* handle,
                  const std::uint32_t* keys,
                  const std::uint32_t* values,
                  std::uint32_t num_keys) {
  using index_type = typename lock_traits<KeyLen, ValueLen>::index_type;
  using key_type = typename lock_traits<KeyLen, ValueLen>::layout_type::key_t;
  using value_type = typename lock_traits<KeyLen, ValueLen>::layout_type::value_t;
  reinterpret_cast<index_type*>(handle->index)->batch_insert(
    reinterpret_cast<key_type*>(const_cast<std::uint32_t*>(keys)),
    reinterpret_cast<value_type*>(const_cast<std::uint32_t*>(values)),
    num_keys);
}

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void erase_typed(lock_handle* handle,
                 const std::uint32_t* keys,
                 std::uint32_t num_keys) {
  using index_type = typename lock_traits<KeyLen, ValueLen>::index_type;
  using key_type = typename lock_traits<KeyLen, ValueLen>::layout_type::key_t;
  reinterpret_cast<index_type*>(handle->index)->batch_delete(
    reinterpret_cast<key_type*>(const_cast<std::uint32_t*>(keys)), nullptr, num_keys);
}

template <std::uint32_t KeyLen, std::uint32_t ValueLen>
void find_typed(lock_handle* handle,
                const std::uint32_t* keys,
                std::uint32_t* results,
                std::uint32_t num_keys) {
  using index_type = typename lock_traits<KeyLen, ValueLen>::index_type;
  using key_type = typename lock_traits<KeyLen, ValueLen>::layout_type::key_t;
  using value_type = typename lock_traits<KeyLen, ValueLen>::layout_type::value_t;
  reinterpret_cast<index_type*>(handle->index)->batch_search(
    reinterpret_cast<key_type*>(const_cast<std::uint32_t*>(keys)),
    reinterpret_cast<value_type*>(results),
    num_keys);
}

[[noreturn]] void unsupported_lengths(std::uint32_t keylen_max, std::uint32_t valuelen_max) {
  std::fprintf(stderr, "Unsupported DyCuckoo lock key/value lengths: %u/%u.\n", keylen_max, valuelen_max);
  std::abort();
}

#define DISPATCH_DYCUCKOO_LOCK_LENGTHS(keylen_max, valuelen_max, func, ...) \
  switch (keylen_max) { \
    case 1: \
      switch (valuelen_max) { \
        case 1: return func<1, 1>(__VA_ARGS__); \
        case 2: return func<1, 2>(__VA_ARGS__); \
        case 4: return func<1, 4>(__VA_ARGS__); \
        case 8: return func<1, 8>(__VA_ARGS__); \
        case 16: return func<1, 16>(__VA_ARGS__); \
      } \
      break; \
    case 2: if (valuelen_max == 1) return func<2, 1>(__VA_ARGS__); break; \
    case 4: if (valuelen_max == 1) return func<4, 1>(__VA_ARGS__); break; \
    case 8: if (valuelen_max == 1) return func<8, 1>(__VA_ARGS__); break; \
    case 16: if (valuelen_max == 1) return func<16, 1>(__VA_ARGS__); break; \
  } \
  unsupported_lengths(keylen_max, valuelen_max)

void* create_index_for_lengths(std::uint32_t keylen_max,
                               std::uint32_t valuelen_max,
                               std::uint32_t init_kv_num,
                               int small_batch_size,
                               double lower_bound,
                               double upper_bound) {
  DISPATCH_DYCUCKOO_LOCK_LENGTHS(
    keylen_max, valuelen_max, create_typed_index, init_kv_num, small_batch_size, lower_bound, upper_bound);
}

void destroy_index_for_lengths(std::uint32_t keylen_max, std::uint32_t valuelen_max, void* index) {
  DISPATCH_DYCUCKOO_LOCK_LENGTHS(keylen_max, valuelen_max, destroy_typed_index, index);
}

void insert_for_lengths(std::uint32_t keylen_max,
                        std::uint32_t valuelen_max,
                        lock_handle* handle,
                        const std::uint32_t* keys,
                        const std::uint32_t* values,
                        std::uint32_t num_keys) {
  DISPATCH_DYCUCKOO_LOCK_LENGTHS(keylen_max, valuelen_max, insert_typed, handle, keys, values, num_keys);
}

void erase_for_lengths(std::uint32_t keylen_max,
                       std::uint32_t valuelen_max,
                       lock_handle* handle,
                       const std::uint32_t* keys,
                       std::uint32_t num_keys) {
  DISPATCH_DYCUCKOO_LOCK_LENGTHS(keylen_max, valuelen_max, erase_typed, handle, keys, num_keys);
}

void find_for_lengths(std::uint32_t keylen_max,
                      std::uint32_t valuelen_max,
                      lock_handle* handle,
                      const std::uint32_t* keys,
                      std::uint32_t* results,
                      std::uint32_t num_keys) {
  DISPATCH_DYCUCKOO_LOCK_LENGTHS(keylen_max, valuelen_max, find_typed, handle, keys, results, num_keys);
}

}  // namespace

extern "C" void* gpu_dycuckoo_dynamic_lock_create(std::uint32_t init_kv_num,
                                                  int small_batch_size,
                                                  double lower_bound,
                                                  double upper_bound,
                                                  std::uint32_t keylen_max,
                                                  std::uint32_t valuelen_max) {
  auto* handle = new lock_handle;
  handle->keylen_max = keylen_max;
  handle->valuelen_max = valuelen_max;
  handle->index = create_index_for_lengths(
    keylen_max, valuelen_max, init_kv_num, small_batch_size, lower_bound, upper_bound);
  return handle;
}

extern "C" void gpu_dycuckoo_dynamic_lock_destroy(void* index) {
  auto* handle = reinterpret_cast<lock_handle*>(index);
  destroy_index_for_lengths(handle->keylen_max, handle->valuelen_max, handle->index);
  delete handle;
}

extern "C" void gpu_dycuckoo_dynamic_lock_insert(void* index,
                                                 const std::uint32_t* keys,
                                                 const std::uint32_t* values,
                                                 std::uint32_t num_keys) {
  auto* handle = reinterpret_cast<lock_handle*>(index);
  insert_for_lengths(handle->keylen_max, handle->valuelen_max, handle, keys, values, num_keys);
}

extern "C" void gpu_dycuckoo_dynamic_lock_erase(void* index,
                                                const std::uint32_t* keys,
                                                std::uint32_t num_keys) {
  auto* handle = reinterpret_cast<lock_handle*>(index);
  erase_for_lengths(handle->keylen_max, handle->valuelen_max, handle, keys, num_keys);
}

extern "C" void gpu_dycuckoo_dynamic_lock_find(void* index,
                                               const std::uint32_t* keys,
                                               std::uint32_t* results,
                                               std::uint32_t num_keys) {
  auto* handle = reinterpret_cast<lock_handle*>(index);
  find_for_lengths(handle->keylen_max, handle->valuelen_max, handle, keys, results, num_keys);
}

#undef DISPATCH_DYCUCKOO_LOCK_LENGTHS
