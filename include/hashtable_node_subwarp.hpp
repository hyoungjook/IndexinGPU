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
#include <cstdint>
#include <macros.hpp>
#include <utils.hpp>
#include <suffix_node_subwarp.hpp>

template <typename tile_type, typename allocator_type>
struct hashtable_node_subwarp {
  using key_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  struct __align__(8) elem_type {
    key_type key;
    value_type value;
  };
  using elem_unsigned_type = uint64_t;
  static constexpr int node_width = 16;
  static constexpr int capacity = node_width - 2;
  static constexpr uint32_t KEYSTATE_VALUE = 0b00u;
  static constexpr uint32_t KEYSTATE_SUFFIX = 0b11u;
  static constexpr uint32_t KEYSTATE_LONGVAL = 0b10u;
  static_assert(tile_type::size() == node_width);
  DEVICE_QUALIFIER hashtable_node_subwarp(const tile_type& tile, allocator_type& allocator)
      : tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER hashtable_node_subwarp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index), tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER void initialize_empty(bool is_head, size_type local_depth = 0, bool is_locked = false) {
    lane_elem_ = {0, 0};
    metadata_ = (
      (0u << num_keys_offset_) |  // num_keys = 0;
      (0u & next_bit_mask_) |     // has_next = false;
      (0u & garbage_bit_mask_)    // is_garbage = false;
    );
    if (is_head) { metadata_ |= head_bit_mask_; }
    if (is_locked) { metadata_ |= lock_bit_mask_; }
    metadata_ |= (local_depth << local_depth_bits_offset_);
    write_metadata_to_registers();
  }

  template <utils::memory_order order>
  DEVICE_QUALIFIER void load_from_array(key_type* table_ptr) {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_));
    do_load<order>(node_ptr);
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void load_from_allocator() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_load<order>(node_ptr);
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void do_load(elem_unsigned_type* node_ptr) {
    auto elem = utils::memory::cacheline_atomic_load<elem_unsigned_type, order>(node_ptr, tile_);
    lane_elem_ = *reinterpret_cast<elem_type*>(&elem);
    read_metadata_from_registers();
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void store_to_array(key_type* table_ptr) {
    auto node_ptr = table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_);
    do_store<order>(reinterpret_cast<elem_unsigned_type*>(node_ptr));
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void store_to_allocator() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_store<order>(node_ptr);
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void store_head_to_array_aux_to_allocator(key_type* table_ptr) {
    auto node_ptr = is_head() ?
        reinterpret_cast<elem_unsigned_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_)) :
        reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_store<order>(node_ptr);
  }
  template <utils::memory_order order>
  DEVICE_QUALIFIER void do_store(elem_unsigned_type* node_ptr) {
    utils::memory::cacheline_atomic_store<elem_unsigned_type, order>(
      node_ptr, *reinterpret_cast<elem_unsigned_type*>(&lane_elem_), tile_);
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = tile_.shfl(lane_elem_.key, metadata_lane_);
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_.key = metadata_;
    }
  }

  DEVICE_QUALIFIER key_type get_key_from_location(const int location) const {
    return tile_.shfl(lane_elem_.key, location);
  }
  DEVICE_QUALIFIER value_type get_value_from_location(const int location) const {
    return tile_.shfl(lane_elem_.value, location);
  }
  DEVICE_QUALIFIER uint32_t get_keystate_from_location(int location) const {
    uint32_t keystate_bits = tile_.shfl(lane_elem_.value, keystate_bits_lane_);
    if (static_cast<uint32_t>(location) < max_num_keys_) {
      return (keystate_bits >> (2 * location)) & keystate_mask_;
    }
    else { // prevent undefiend shifts
      return 0;
    }
  }
  static DEVICE_QUALIFIER bool keystate_has_more_key(const uint32_t keystate) {
    return (keystate & keystate_mask_more_key_) != 0;
  }
  static DEVICE_QUALIFIER bool keystate_has_suffix_ptr(const uint32_t keystate) {
    return (keystate & keystate_mask_suffix_) != 0;
  }
  DEVICE_QUALIFIER bool is_valid_lane() const {
    return tile_.thread_rank() < num_keys();
  }

  DEVICE_QUALIFIER uint32_t num_keys() const {
    return (metadata_ & num_keys_mask_) >> num_keys_offset_;
  }
  DEVICE_QUALIFIER void set_num_keys(const uint32_t& value) {
    assert(value <= (num_keys_mask_ >> num_keys_offset_));
    metadata_ &= ~num_keys_mask_;
    metadata_ |= (value << num_keys_offset_);
  }
  DEVICE_QUALIFIER bool is_full() const {
    return (num_keys() == max_num_keys_);
  }
  DEVICE_QUALIFIER bool is_mergeable(const hashtable_node_subwarp& next_node) const {
    return (num_keys() + next_node.num_keys()) <= max_num_keys_;
  }
  DEVICE_QUALIFIER bool has_next() const {
    return static_cast<bool>(metadata_ & next_bit_mask_);
  }
  DEVICE_QUALIFIER void set_has_next() {
    metadata_ |= next_bit_mask_;
    write_metadata_to_registers();
  }
  DEVICE_QUALIFIER value_type get_next_index() const {
    return tile_.shfl(lane_elem_.value, next_ptr_lane_);
  }
  DEVICE_QUALIFIER void set_next_index(const value_type& index) {
    if (tile_.thread_rank() == next_ptr_lane_) { lane_elem_.value = index; }
  }
  DEVICE_QUALIFIER bool is_head() const {
    return static_cast<bool>(metadata_ & head_bit_mask_);
  }
  DEVICE_QUALIFIER bool is_garbage() const {
    return static_cast<bool>(metadata_ & garbage_bit_mask_);
  }
  DEVICE_QUALIFIER void make_garbage() {
    metadata_ |= garbage_bit_mask_;
    write_metadata_to_registers();
  }
  template <utils::memory_order order>
  static DEVICE_QUALIFIER bool is_garbage(size_type node_index, const tile_type& tile, allocator_type& allocator) {
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(node_index));
    key_type metadata;
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (order == utils::memory_order::acq_rel ||
                    order == utils::memory_order::relaxed) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata = metadata_ref.load(order == utils::memory_order::acq_rel ? 
          cuda::memory_order_acquire : cuda::memory_order_relaxed);
      }
      else {
        metadata = bucket_ptr[metadata_lane_].key;
      }
    }
    return static_cast<bool>(tile.shfl(metadata, metadata_lane_) & garbage_bit_mask_);
  }
  template <utils::memory_order order>
  static DEVICE_QUALIFIER void make_garbage(size_type node_index, const tile_type& tile, allocator_type& allocator) {
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(node_index));
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (order == utils::memory_order::acq_rel ||
                    order == utils::memory_order::relaxed) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata_ref.fetch_or(garbage_bit_mask_,
          order == utils::memory_order::acq_rel ? cuda::memory_order_release :
                                                  cuda::memory_order_relaxed);
      }
      else {
        bucket_ptr[metadata_lane_].key |= garbage_bit_mask_;
      }
    }
  }
  DEVICE_QUALIFIER size_type get_local_depth() const {
    return ((metadata_ & local_depth_bits_mask_) >> local_depth_bits_offset_);
  }
  DEVICE_QUALIFIER void set_local_depth(size_type local_depth) {
    metadata_ &= ~local_depth_bits_mask_;
    metadata_ |= (local_depth << local_depth_bits_offset_);
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER size_type get_node_index() const { return node_index_; }

  // lock/unlock for head nodes embedded in array (chainHT, cuckooHT)
  static DEVICE_QUALIFIER bool try_lock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    key_type old;
    auto bucket_ptr = reinterpret_cast<elem_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * bucket_index));
    if (tile.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
      old = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    // if previously not locked, now it's locked
    bool is_locked = (tile.shfl(old, metadata_lane_) & lock_bit_mask_) == 0;
    if (is_locked) { cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device); }
    return is_locked;
  }
  static DEVICE_QUALIFIER void lock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    while (!try_lock(table_ptr, bucket_index, tile));
  }
  template <bool release = true>
  static DEVICE_QUALIFIER void unlock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    // unlock, only using the pointer, not load the entire register
    auto bucket_ptr = reinterpret_cast<elem_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * bucket_index));
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (release) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
      }
      else {
        bucket_ptr[metadata_lane_].key &= ~lock_bit_mask_;
      }
    }
  }
  // lock/unlock for head nodes in slab allocator (extendHT)
  static DEVICE_QUALIFIER bool try_lock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    key_type old;
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(head_index));
    if (tile.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
      old = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    // if previously not locked, now it's locked
    bool is_locked = (tile.shfl(old, metadata_lane_) & lock_bit_mask_) == 0;
    if (is_locked) { cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device); }
    return is_locked;
  }
  static DEVICE_QUALIFIER void lock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    while (!try_lock(head_index, tile, allocator));
  }
  template <bool release = true>
  static DEVICE_QUALIFIER void unlock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    // unlock, only using the pointer, not load the entire register
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(head_index));
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (release) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
      }
      else {
        bucket_ptr[metadata_lane_].key &= ~lock_bit_mask_;
      }
    }
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool more_key) const {
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    return tile_.ballot(is_valid_lane() &&
                        lane_elem_.key == key &&
                        keystate_has_more_key(lane_keystate) == more_key);
  }
  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, uint32_t keystate) const {
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    return tile_.ballot(is_valid_lane() &&
                        lane_elem_.key == key &&
                        lane_keystate == keystate);
  }
  DEVICE_QUALIFIER uint32_t match_key_value_in_node(const key_type& key, const value_type& value, uint32_t keystate) const {
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    return tile_.ballot(is_valid_lane() &&
                        lane_elem_.key == key &&
                        lane_elem_.value == value &&
                        lane_keystate == keystate);
  }

  DEVICE_QUALIFIER void insert(const key_type& key, const value_type& value, uint32_t keystate) {
    assert(!is_full());
    auto location = num_keys();
    if (tile_.thread_rank() == location) {
      lane_elem_ = {key, value};
    }
    if (tile_.thread_rank() == keystate_bits_lane_) {
      const int key_lane_x2 = 2 * location;
      lane_elem_.value &= ((1u << key_lane_x2) - 1);
      lane_elem_.value |= ((keystate & keystate_mask_) << key_lane_x2);
    }
    metadata_++;    // equiv. to num_keys++
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void update(int location, const value_type& value, uint32_t keystate) {
    assert(location < num_keys());
    if (tile_.thread_rank() == location) {
      lane_elem_.value = value;
    }
    if (tile_.thread_rank() == keystate_bits_lane_) {
      auto key_location_x2 = 2 * location;
      lane_elem_.value &= ~(keystate_mask_ << key_location_x2);
      lane_elem_.value |= ((keystate & keystate_mask_) << key_location_x2);
    }
  }

  DEVICE_QUALIFIER void merge(hashtable_node_subwarp<tile_type, allocator_type>& next_node) {
    assert(is_mergeable(next_node));
    // copy elements from next node
    elem_type shifted_elem;
    auto this_num_keys = num_keys();
    shifted_elem.key = tile_.shfl_up(next_node.lane_elem_.key, this_num_keys);
    shifted_elem.value = tile_.shfl_up(next_node.lane_elem_.value, this_num_keys);
    auto new_num_keys = this_num_keys + next_node.num_keys();
    if (this_num_keys <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) {
      lane_elem_ = shifted_elem;
    }
    if (tile_.thread_rank() == keystate_bits_lane_) {
      auto num_keys_x2 = 2 * this_num_keys;
      uint32_t num_keys_x2_mask = ((1u << num_keys_x2) - 1);
      lane_elem_.value &= num_keys_x2_mask;
      lane_elem_.value |= ((next_node.lane_elem_.value << num_keys_x2) & ~num_keys_x2_mask);
    }
    set_num_keys(new_num_keys);
    // this node's next is next node's next
    // next node's next is this node
    if (tile_.thread_rank() == next_ptr_lane_) {
      lane_elem_.value = next_node.lane_elem_.value;
      next_node.lane_elem_.value = node_index_;
    }
    // has_next = next.has_next
    // next.has_next = true; next.make_garbage()
    metadata_ &= ~next_bit_mask_;
    metadata_ |= (next_node.metadata_ & next_bit_mask_);
    next_node.metadata_ |= (next_bit_mask_ | garbage_bit_mask_);
    write_metadata_to_registers();
    next_node.write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void erase(int location) {
    assert(location < num_keys());
    metadata_--;    // equiv. to num_keys--
    elem_type down_elem;
    down_elem.key = tile_.shfl_down(lane_elem_.key, 1);
    down_elem.value = tile_.shfl_down(lane_elem_.value, 1);
    if (is_valid_lane()) {
      if (tile_.thread_rank() >= location) {
        lane_elem_ = down_elem;
      }
    }
    if (tile_.thread_rank() == keystate_bits_lane_) {
      uint32_t key_lane_x2_mask = ((1u << (2 * location)) - 1);
      lane_elem_.value = (lane_elem_.value & key_lane_x2_mask) |
                         ((lane_elem_.value >> 2) & ~key_lane_x2_mask);
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER hashtable_node_subwarp<tile_type, allocator_type>& operator=(
        const hashtable_node_subwarp<tile_type, allocator_type>& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u]: {", node_index_);
    if (num_keys() > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    }
    if (num_keys() == 0) {
      if (lead_lane) printf("empty}\n");
      return;
    }
    if (is_head()) {
      if (lead_lane) printf("head ");
    }
    if (is_garbage()) {
      if (lead_lane) printf("garbage ");
    }
    if (lead_lane) printf("ld(%u) ", get_local_depth());
    if (lead_lane) printf("%u ", num_keys());
    for (size_type i = 0; i < num_keys(); ++i) {
      key_type key = get_key_from_location(i);
      value_type value = get_value_from_location(i);
      uint32_t keystate = get_keystate_from_location(i);
      auto keystate_symbol = [](uint32_t keystate) -> const char* {
        if (keystate == KEYSTATE_VALUE) { return "$"; }
        if (keystate == KEYSTATE_SUFFIX) { return "s"; }
        if (keystate == KEYSTATE_LONGVAL) { return "v"; }
        return "?";
      };
      if (lead_lane) printf("(%u %u %s) ", key, value, keystate_symbol(keystate));
    }
    if (lead_lane) printf("%s ", (metadata_ & lock_bit_mask_) ? "locked" : "free");
    size_type next_index = get_next_index();
    if (has_next()) {
      if (lead_lane) printf("next(%u)", next_index);
    }
    else {
      if (lead_lane) printf("nullnext");
    }
    if (lead_lane) printf("}\n");
    for (size_type i = 0; i < num_keys(); ++i) {
      uint32_t keystate = get_keystate_from_location(i);
      if (keystate_has_suffix_ptr(keystate)) {
        size_type suffix_index = get_value_from_location(i);
        auto suffix = suffix_node_subwarp<tile_type, allocator_type>(suffix_index, tile_, allocator_);
        suffix.load_head();
        suffix.print();
      }
    }
  }

 private:
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;

  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key,val0] [key,val1] ... [key,val13] | [-,keystates] [metadata,next]

  // keystates: each key 13-0 has 2 bits (total 28 bits used)
  //      (MSB)[empty:4][ks13:2][ks12:2]...[ks1:2][ks0:2](LSB)
  //      Each (key, value) pair has three options:
  //        (1) keystate=0b00 (key_suffix=0, key_more=0): value is the final value, key ends here
  //        (2) keystate=0b11 (key_suffix=1, key_more=1): value is link to the suffix info, key continues (but suffix info contains the final value)
  //        (3) keystate=0b10 (key_suffix=1, key_more=0): value is link to the suffix info but key ends here (final value is long)

  // metadata is 32bits.
  //    (MSB)
  //    [empty:18]
  //    [local_depth:6]
  //    [is_garbage:1][is_head:1]
  //    [has_next:1][is_locked:1]
  //    [num_keys:4]
  //    (LSB)
  //  - is_garbage, is_locked are only valid for head nodes
  //  - local_depth for the node chain is all same

  static_assert(sizeof(elem_type) == sizeof(uint64_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;  // key
  static constexpr uint32_t next_ptr_lane_ = node_width - 1;  // value
  static constexpr uint32_t keystate_bits_lane_ = node_width - 2; // value
  static constexpr uint32_t num_keys_offset_ = 0;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t num_keys_mask_ = ((1u << num_keys_bits_) - 1) << num_keys_offset_;
  static constexpr uint32_t lock_bit_offset_ = 4;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t next_bit_offset_ = 5;
  static constexpr uint32_t next_bit_mask_ = 1u << next_bit_offset_;
  static constexpr uint32_t head_bit_offset_ = 6;
  static constexpr uint32_t head_bit_mask_ = 1u << head_bit_offset_;
  static constexpr uint32_t garbage_bit_offset_ = 7;
  static constexpr uint32_t garbage_bit_mask_ = 1u << garbage_bit_offset_;
  static constexpr uint32_t local_depth_bits_offset_ = 8;
  static constexpr uint32_t local_depth_bits_bits_ = 6;
  static constexpr uint32_t local_depth_bits_mask_ = ((1u << local_depth_bits_bits_) - 1) << local_depth_bits_offset_;
  static constexpr uint32_t max_num_keys_ = node_width - 2;
  static_assert(num_keys_offset_ == 0); // this allows (metadata +/- N) equivalent to (num_keys +/- N) within range
  static_assert(max_num_keys_ == capacity);

  uint32_t metadata_;
  static constexpr uint32_t keystate_mask_more_key_ = 0b01u;
  static constexpr uint32_t keystate_mask_suffix_ = 0b10u;
  static constexpr uint32_t keystate_mask_ = 0b11u;
};
