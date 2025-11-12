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
#include <macros.hpp>
#include <memory_utils.hpp>
#include <utils.hpp>

template <typename tile_type>
struct masstree_node {
  using elem_type = uint32_t;
  using key_type = elem_type;
  using value_type = elem_type;
  using size_type = uint32_t;
  static constexpr int node_width = 16;
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  DEVICE_QUALIFIER masstree_node(elem_type* ptr, const size_type index, const tile_type& tile)
      : node_ptr_(ptr), node_index_(index), tile_(tile), is_locked_(false) {
    assert(tile_.size() == 2 * node_width);
  }
  DEVICE_QUALIFIER masstree_node(elem_type* ptr,
                                 const size_type index,
                                 const tile_type& tile,
                                 const elem_type elem,
                                 bool is_locked,
                                 bool is_leaf,
                                 uint16_t num_keys,
                                 uint16_t ptr_meta_bits)
      : node_ptr_(ptr)
      , node_index_(index)
      , lane_elem_(elem)
      , tile_(tile)
      , is_locked_(is_locked)
      , is_leaf_(is_leaf)
      , num_keys_(num_keys)
      , ptr_meta_bits_(ptr_meta_bits) {}

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    is_locked_ = false;
    is_leaf_ = true;
    num_keys_ = 0;
    ptr_meta_bits_ = 0;
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_elem_ = cuda_memory<elem_type>::load(node_ptr_ + tile_.thread_rank(), order);
    read_metadata_from_registers();
  }
  DEVICE_QUALIFIER void store(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<elem_type>::store(node_ptr_ + tile_.thread_rank(), lane_elem_, order);
  }

  DEVICE_QUALIFIER int get_key_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return location;
  }
  DEVICE_QUALIFIER int get_value_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return node_width + location;
  }

  DEVICE_QUALIFIER key_type get_key_from_location(const int location) const {
    return tile_.shfl(lane_elem_, get_key_lane_from_location(location));
  }
  DEVICE_QUALIFIER value_type get_value_from_location(const int location) const {
    return tile_.shfl(lane_elem_, get_value_lane_from_location(location));
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys_;
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    // intermediate node has children one more than keys
    return node_width <= tile_.thread_rank() &&
           tile_.thread_rank() < node_width + num_keys_ + (is_leaf_ ? 0 : 1);
  }

  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    assert(!is_leaf_);
    const bool key_greater_equal = is_valid_key_lane() && (key >= lane_elem_);
    uint32_t key_greater_equal_bitmap = tile_.ballot(key_greater_equal);
    auto next_location = utils::bits::bfind(key_greater_equal_bitmap) + 1;
    assert(0 <= next_location && next_location < num_keys_ + 1);
    return next_location;
  }
  DEVICE_QUALIFIER value_type find_next(const key_type& key) const {
    auto next_location = find_next_location(key);
    return tile_.shfl(lane_elem_, get_value_lane_from_location(next_location));
  }
  DEVICE_QUALIFIER bool key_is_in_upperhalf(const key_type& key) const {
    auto key_at_upper_half = get_key_from_location(half_node_width_);
    return (key >= key_at_upper_half);
  }

  DEVICE_QUALIFIER int find_key_location_in_node(const key_type& key) const {
    auto key_exist = tile_.ballot(is_valid_key_lane() && lane_elem_ == key);
    return __ffs(key_exist) - 1;
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key) const {
    auto key_exist = tile_.ballot(is_valid_key_lane() && lane_elem_ == key);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER int find_ptr_location_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() && lane_elem_ == ptr);
    return __ffs(ptr_exist >> node_width) - 1;
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() && lane_elem_ == ptr);
    return (ptr_exist != 0);
  }
  DEVICE_QUALIFIER value_type get_key_value_from_node(const key_type& key) const {
    auto key_location = find_key_location_in_node(key);
    return key_location == -1 ? invalid_value : get_value_from_location(key_location);
  }

  DEVICE_QUALIFIER bool is_leaf() const { return is_leaf_; }
  DEVICE_QUALIFIER bool is_intermediate() const { return !is_leaf_; }
  DEVICE_QUALIFIER bool is_full() const {
    assert(num_keys_ <= max_num_keys_);
    return (num_keys_ == max_num_keys_);
  }
  DEVICE_QUALIFIER bool has_sibling() const { return ptr_meta_bits_ & right_sibling_ptr_meta_mask_; }
  DEVICE_QUALIFIER key_type get_high_key() const {
    return get_key_from_location(sibling_location_);
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return get_value_from_location(sibling_location_);
  }

  template <typename T, uint32_t offset>
  DEVICE_QUALIFIER bool is_bit_set(const T& metadata) {
    return (metadata & (1u << offset)) != 0;
  }
  template <uint32_t offset, uint32_t num_bits>
  DEVICE_QUALIFIER uint32_t extract_bits(const uint32_t& metadata) {
    return (metadata >> offset) & ((1u << num_bits) - 1);
  }
  template <uint32_t offset>
  DEVICE_QUALIFIER void embed_bits(uint32_t& metadata, const uint32_t bits) {
    metadata |= (bits << offset);
  }
  DEVICE_QUALIFIER void read_metadata_from_registers() {
    uint32_t metadata = tile_.shfl(lane_elem_, metadata_lane_);
    is_locked_ = extract_bits<lock_bit_offset_, 1>(metadata);
    is_leaf_ = extract_bits<leaf_bit_offset_, 1>(metadata);
    num_keys_ = extract_bits<num_keys_offset_, num_keys_bits_>(metadata);
    ptr_meta_bits_ = extract_bits<ptr_meta_bits_offset_, node_width>(metadata);
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      uint32_t metadata = 0;
      embed_bits<lock_bit_offset_>(metadata, is_locked_);
      embed_bits<leaf_bit_offset_>(metadata, is_leaf_);
      embed_bits<num_keys_offset_>(metadata, num_keys_);
      embed_bits<ptr_meta_bits_offset_>(metadata, ptr_meta_bits_);
      lane_elem_ = metadata;
    }
  }
  template <uint32_t offset, bool bit>
  DEVICE_QUALIFIER void set_bit_in_metadata() {
    if (tile_.thread_rank() == metadata_lane_) {
      if (bit) {
        lane_elem_ |= (1u << offset);
      }
      else {
        lane_elem_ &= ~(1u << offset);
      }
    }
  }

  DEVICE_QUALIFIER bool is_locked() const { return is_locked_; }
  DEVICE_QUALIFIER bool try_lock() {
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                     static_cast<elem_type>(lock_bit_mask_));
    }
    old = tile_.shfl(old, metadata_lane_);
    is_locked_ = !is_bit_set<elem_type, lock_bit_offset_>(old);
    if (is_locked_) {
      set_bit_in_metadata<lock_bit_offset_, true>();
      __threadfence();
    }
    else {
      set_bit_in_metadata<lock_bit_offset_, false>();
    }
    return is_locked_;
  }
  DEVICE_QUALIFIER void lock() {
    while (auto failed = !try_lock()) {}
    is_locked_ = true;
  }
  DEVICE_QUALIFIER void unlock() {
    __threadfence();
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicAnd(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                      static_cast<elem_type>(~lock_bit_mask_));
    }
    is_locked_ = false;
    set_bit_in_metadata<lock_bit_offset_, false>();
  }

  DEVICE_QUALIFIER masstree_node do_split(const value_type right_sibling_index,
                                          elem_type* right_sibling_ptr,
                                          const bool make_sibling_locked = false) {
    // prepare the upper half in right sibling
    uint16_t shfl_delta = half_node_width_ + (is_leaf_ ? 0 : 1);
    auto right_sibling_elem = tile_.shfl_down(lane_elem_, shfl_delta);
    auto right_sibling_minimum = get_key_from_location(half_node_width_);
    
    // distribute num keys
    uint16_t right_num_keys = num_keys_ - shfl_delta;
    num_keys_ = half_node_width_;
    
    // distribute ptr meta bits for regular children
    // sibling ptr meta bits will be set below
    uint16_t right_ptr_meta_bits = ptr_meta_bits_ >> shfl_delta;

    // reconnect right sibling pointers
    if (tile_.thread_rank() == get_key_lane_from_location(sibling_location_)) {
      // right's highkey = this node's previous highkey
      right_sibling_elem = lane_elem_;
      // left's highkey = pivot key (right's minimum key)
      lane_elem_ = right_sibling_minimum;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(sibling_location_)) {
      // right's right sibling = this node's previous right sibling
      right_sibling_elem = lane_elem_;
      // left's right sibling = right
      lane_elem_ = right_sibling_index;
    }
    // right's right sibling validity inherits this node's previous' sibling validity
    right_ptr_meta_bits &= ~right_sibling_ptr_meta_mask_;
    right_ptr_meta_bits |= (ptr_meta_bits_ & right_sibling_ptr_meta_mask_);
    // left's right sibling validity becomes true (it now has sibling = right!)
    ptr_meta_bits_ |= right_sibling_ptr_meta_mask_;

    if (is_leaf_) {
      // reconnect left sibling pointers
      if (tile_.thread_rank() == get_value_lane_from_location(left_sibling_location_)) {
        // left's left sibling remains the same (this node's previous' left sibling)
        // right's left sibling is left (this)
        right_sibling_elem = node_index_;
        right_ptr_meta_bits |= left_sibling_ptr_meta_mask_;
      }
    }

    // create right sibling node
    masstree_node right_sibling_node =
        masstree_node(right_sibling_ptr, right_sibling_index, tile_, right_sibling_elem, make_sibling_locked, is_leaf_, right_num_keys, right_ptr_meta_bits);

    // flush metadata
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();

    return right_sibling_node;
  }

  struct split_intermediate_result {
    masstree_node parent;
    masstree_node sibling;
  };
  // Note parent must be locked before this gets called
  DEVICE_QUALIFIER split_intermediate_result split(const value_type right_sibling_index,
                                                   const value_type parent_index,
                                                   elem_type* right_sibling_ptr,
                                                   elem_type* parent_ptr,
                                                   const bool make_sibling_locked = false) {
    // We assume here that the parent is locked
    
    // get pivot_key here in case it's cleared in do_split() for intermediate node
    auto pivot_key = get_key_from_location(half_node_width_);
    auto split_result = do_split(right_sibling_index, right_sibling_ptr, make_sibling_locked);

    // Update parent
    auto parent_node = masstree_node(parent_ptr, parent_index, tile_);
    parent_node.load(cuda_memory_order::memory_order_relaxed);

    // update the parent
    parent_node.insert(pivot_key, right_sibling_index);
    return {parent_node, split_result};
  }

  struct two_nodes_result {
    masstree_node left;
    masstree_node right;
  };
  DEVICE_QUALIFIER two_nodes_result split_as_root(const value_type left_sibling_index,
                                                  const value_type right_sibling_index,
                                                  elem_type* left_sibling_ptr,
                                                  elem_type* right_sibling_ptr,
                                                  const bool make_children_locked = false) {
    // Create a new root
    auto right_node_minimum = get_key_from_location(half_node_width_);

    // Copy the current node into a child
    auto left_child =
        masstree_node(left_sibling_ptr, left_sibling_index, tile_, lane_elem_, 
                      make_children_locked, is_leaf_, num_keys_, ptr_meta_bits_);
    // if the root was a leaf, now it should be intermediate
    if (is_leaf_) { is_leaf_ = false; }
    // Make new root
    num_keys_ = 1;
    if (tile_.thread_rank() == get_key_lane_from_location(0)) {
      lane_elem_ = right_node_minimum;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(0)) {
      lane_elem_ = left_sibling_index;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(1)) {
      lane_elem_ = right_sibling_index;
    }
    ptr_meta_bits_ = 0; // root has no sibling
    write_metadata_to_registers();

    // now split the left child
    auto right_child =
        left_child.do_split(right_sibling_index, right_sibling_ptr, make_children_locked);
    return {left_child, right_child};
  }

  DEVICE_QUALIFIER bool scatter_ptr_meta_bits() {
    int value_location = static_cast<int>(tile_.thread_rank()) - node_width;
    if (value_location >= 0) {
      return (ptr_meta_bits_ >> value_location) & 1u;
    }
    return false;
  }

  DEVICE_QUALIFIER void gather_ptr_meta_bits(bool bit) {
    ptr_meta_bits_ = static_cast<uint16_t>(tile_.ballot(bit) >> node_width);
  }

  DEVICE_QUALIFIER bool insert(const key_type& key, const value_type& value) {
    auto key_location = find_key_location_in_node(key);
    // if the key exists, we update the valueget_key_value_from_node
    if (key_location >= 0) {
      assert(is_leaf_);
      if (tile_.thread_rank() == get_value_lane_from_location(key_location)) {
        lane_elem_ = value;
      }
      // TODO modify ptr_meta_bits_
      //ptr_meta_bits_ |= (value_meta_bit << key_location);
      return false;
    }
    else {
      // else we shuffle the keys and do the insertion
      assert(!is_full());
      const bool key_is_larger = is_valid_key_lane() && (key > lane_elem_);
      uint32_t key_is_larger_bitmap = tile_.ballot(key_is_larger);
      auto key_location = utils::bits::bfind(key_is_larger_bitmap) + 1;
      auto value_location = key_location + (is_leaf_ ? 0 : 1);
      
      ++num_keys_;
      const int key_lane = get_key_lane_from_location(key_location);
      const int value_lane = get_value_lane_from_location(value_location);
      auto up_elem = tile_.shfl_up(lane_elem_, 1);
      bool ptr_meta_bit = scatter_ptr_meta_bits();
      bool up_ptr_meta_bit = tile_.shfl_up(ptr_meta_bit, 1);
      if (is_valid_key_lane()) {
        if (tile_.thread_rank() == key_lane) { lane_elem_ = key; }
        else if (tile_.thread_rank() > key_lane) { lane_elem_ = up_elem; }
      }
      else if (is_valid_value_lane()) {
        if (tile_.thread_rank() == value_lane) {
          lane_elem_ = value;
          // TODO modify ptr_meta_bits_
          //ptr_meta_bit = value_meta_bit;
        }
        else if (tile_.thread_rank() > value_lane) {
          lane_elem_ = up_elem;
          ptr_meta_bit = up_ptr_meta_bit;
        }
      }
      gather_ptr_meta_bits(ptr_meta_bit);
      write_metadata_to_registers();
      return true;
    }
  }

  DEVICE_QUALIFIER bool erase(const key_type& key) {
    // check if key exists
    auto key_location = find_key_location_in_node(key);
    if (key_location >= 0) {
      assert(num_keys_ > 0);
      --num_keys_;
      const int key_lane = get_key_lane_from_location(key_location);
      const int value_lane = get_value_lane_from_location(key_location) + (is_leaf_ ? 0 : 1);
      auto down_elem = tile_.shfl_down(lane_elem_, 1);
      bool ptr_meta_bit = scatter_ptr_meta_bits();
      bool down_ptr_meta_bit = tile_.shfl_down(ptr_meta_bit, 1);
      if (is_valid_key_lane()) {
        if (tile_.thread_rank() >= key_lane) { lane_elem_ = down_elem; }
      }
      else if (is_valid_value_lane()) {
        if (tile_.thread_rank() >= value_lane) {
          lane_elem_ = down_elem;
          ptr_meta_bit = down_ptr_meta_bit;
        }
      }
      gather_ptr_meta_bits(ptr_meta_bit);
      write_metadata_to_registers();
      return true;
    }
    else {
      return false;
    }
  }

  DEVICE_QUALIFIER masstree_node<tile_type>& operator=(
      const masstree_node<tile_type>& other) {
    node_ptr_ = other.node_ptr_;
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    is_locked_ = other.is_locked_;
    is_leaf_ = other.is_leaf_;
    num_keys_ = other.num_keys_;
    ptr_meta_bits_ = other.ptr_meta_bits_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u](%p): {", node_index_, node_ptr_);
    if (num_keys_ > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    } 
    for (size_type i = 0; i < num_keys_; ++i) {
      elem_type key = tile_.shfl(lane_elem_, get_key_lane_from_location(i));
      elem_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
      if (lead_lane) printf("(%u %u) ", key, value);
    }
    if (!is_leaf_) {
      elem_type last_value = tile_.shfl(lane_elem_, get_value_lane_from_location(num_keys_));
      if (lead_lane) printf("(x %u) ", last_value);
    }
    if (lead_lane) printf("%s %s ", is_locked_ ? "lckd" : "free", is_leaf_ ? "leaf" : "intr");
    elem_type sibling_key = get_high_key();
    elem_type sibling_index = get_sibling_index();
    if (has_sibling() && lead_lane) printf("rsbl(%u %u)", sibling_key, sibling_index);
    if (lead_lane) printf("}\n");
  }

 private:
  elem_type* node_ptr_;
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type tile_;
  
  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key0] [key1] ... [key13] | [metadata] | [key15]
  //  [ptr0] [ptr1] ... [ptr13] | [ptr14]    | [ptr15]
  // note key14 doesn't exist; it's metadata.
  // ptr14 is:
  //    for intermediate node, just a child pointer (b/c #children = #key + 1)
  //    for leaf node, left sibling pointer
  // key15 is high key = right sibling's minimum key.
  // ptr15 is right sibling pointer

  // metadata is 32bits.
  // MSB - [ptr_meta_bits:16][empty:10][num_keys:4][is_leaf:1][is_locked:1] - LSB
  //    ptr_meta_bits: each corresponds to ptr15-ptr0 (MSB-LSB).
  //      for sibling pointers: bit=0 means NULL (sibling not exists)
  //      for intermediate node's child pointers: unused
  //      for leaf (border) node's child pointers: bit=0 (leaf pointer), bit=1 (root node pointer to next masstree layer)
  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t metadata_lane_ = node_width - 2;
  
  static constexpr uint32_t lock_bit_offset_ = 0;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t leaf_bit_offset_ = 1;
  static constexpr uint32_t num_keys_offset_ = 2;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t ptr_meta_bits_offset_ = 16;

  static constexpr uint16_t left_sibling_ptr_meta_mask_ = 1u << 14;
  static constexpr uint16_t right_sibling_ptr_meta_mask_ = 1u << 15;

  static constexpr size_type max_num_keys_ = node_width - 2;
  static constexpr uint32_t sibling_location_ = node_width - 1;
  static constexpr uint32_t left_sibling_location_ = node_width - 2;
  static constexpr uint32_t half_node_width_ = max_num_keys_ >> 1;

  bool is_locked_;
  bool is_leaf_;
  uint16_t num_keys_;
  uint16_t ptr_meta_bits_;
};

template <typename btree>
__global__ void print_tree_nodes_kernel(btree tree) {
  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<btree::cg_tile_size>(block);
  tree.print_tree_nodes_device_func(tile);
}
