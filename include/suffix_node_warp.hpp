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
#include <cstdint>
#include <macros.hpp>
#include <utils.hpp>

template <typename tile_type, typename allocator_type>
struct suffix_node_warp {
  using elem_type = uint32_t;
  using size_type = uint32_t;
  static constexpr int node_width = 32;
  static_assert(tile_type::size() == node_width);
  DEVICE_QUALIFIER suffix_node_warp(const tile_type& tile, allocator_type& allocator): tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER suffix_node_warp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index)
      , tile_(tile)
      , allocator_(allocator) {}
  
  // ALL suffix loads/stores in this file are done as non-atomic, because
  //  - suffix loads are done with the pointer in tree/bucket node, which is loaded with memory_order_acquire
  //  - suffix stores are protected by tree/bucket node's locks, which includes threadfence with memory_order_release
  DEVICE_QUALIFIER void load_head() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    lane_elem_ = utils::memory::load<elem_type, false>(node_ptr + tile_.thread_rank());
  }
  DEVICE_QUALIFIER void store_head() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    utils::memory::store<elem_type, false>(node_ptr + tile_.thread_rank(), lane_elem_);
  }

  DEVICE_QUALIFIER size_type get_node_index() const {
    return node_index_;
  }

  DEVICE_QUALIFIER size_type get_next() const {
    return tile_.shfl(lane_elem_, next_lane_);
  }
  DEVICE_QUALIFIER size_type get_key_length() const {
    auto lengths = tile_.shfl(lane_elem_, head_node_length_lane_);
    return (lengths >> key_length_offset_bits_) & length_mask_;
  }
  DEVICE_QUALIFIER size_type get_value_length() const {
    auto lengths = tile_.shfl(lane_elem_, head_node_length_lane_);
    return (lengths >> value_length_offset_bits_) & length_mask_;
  }

  DEVICE_QUALIFIER uint32_t get_num_nodes() const {
    auto total_length = get_key_length() + get_value_length();
    // first one element is length, so (total_length + 1)
    return ((total_length + 1) + node_max_len_ - 1) / node_max_len_;
  }

  template <typename keyptr_or_keystore>
  DEVICE_QUALIFIER bool streq(keyptr_or_keystore key, uint32_t key_length) const {
    if (get_key_length() != key_length) { return false; }
    // now key_length == this_key_length, compare head node
    // ignore first one element in head
    int skip_elems = 1;
    int key_offset = -skip_elems;
    key_length += skip_elems;
    auto elem = lane_elem_;
    while (true) {
      bool mismatch = (skip_elems <= tile_.thread_rank()) &&
                      (tile_.thread_rank() < node_max_len_) &&
                      (tile_.thread_rank() < key_length) && 
                      (elem != key[key_offset + tile_.thread_rank()]);
      uint32_t mismatch_ballot = tile_.ballot(mismatch);
      if (mismatch_ballot != 0) { return false; }
      if (key_length <= node_max_len_) { return true; }
      key_offset += node_max_len_;
      key_length -= node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
    assert(false);
  }

  template <typename keyptr_or_keystore>
  DEVICE_QUALIFIER int strcmp(keyptr_or_keystore key, uint32_t key_length, elem_type* mismatch_keyslice = nullptr) const {
    // strcmp(this, key) -> 0 (match), +(this<key), -(this>key)
    // the absolute of return value: (1 + num_matches)
    // NOTE if one is prefix of the other, num_matches is (len(smaller) - 1)
    auto this_length = get_key_length();
    auto cmp_length = min(this_length, key_length);
    auto elem = lane_elem_;
    int total_num_matches = 0;
    // ignore first one element in head
    int skip_elems = 1;
    int key_offset = -skip_elems;
    key_length += skip_elems;
    this_length += skip_elems;
    cmp_length += skip_elems;
    while (true) {
      // compare elements
      bool this_more = (tile_.thread_rank() < this_length - 1);
      bool key_more = (tile_.thread_rank() < key_length - 1);
      bool valid_cmp = (skip_elems <= tile_.thread_rank()) &&
                       (tile_.thread_rank() < node_max_len_) &&
                       (tile_.thread_rank() < cmp_length);
      elem_type other = valid_cmp ? (key[key_offset + tile_.thread_rank()]) : 0;
      bool match = valid_cmp &&
                   (elem == other) &&
                   (this_more == key_more);
      match = match || (tile_.thread_rank() < skip_elems);
      uint32_t match_ballot = tile_.ballot(match);
      int num_matches = __ffs(~match_ballot) - 1;
      // if all elements match
      if (num_matches == cmp_length) { return 0; }
      // if found mismatch in this node
      total_num_matches += (num_matches - skip_elems);
      if (num_matches < node_max_len_ && num_matches < cmp_length) {
        // num_matches'th lane has the first mismatch elems
        bool ge = tile_.shfl((elem < other) || (elem == other && this_more < key_more), num_matches);
        if (mismatch_keyslice) { *mismatch_keyslice = tile_.shfl(elem, num_matches); }
        return (ge ? 1 : -1) * static_cast<int>(1 + total_num_matches);
      }
      // proceed to next node
      cmp_length -= node_max_len_;
      this_length -= node_max_len_;
      key_length -= node_max_len_;
      key_offset += node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
    assert(false);
  }

  template <uint32_t prime0>
  DEVICE_QUALIFIER uint32_t compute_polynomial() const {
    // compute (1 * s[0]) + (p * s[1]) + (p^2 * s[2]) + ... + (p^(l-1) * s[l-1])
    // 1. exponent = [1, p, p^2, ..., p^31]
    uint32_t exponent0 = (tile_.thread_rank() == 0) ? 1 : prime0;
    for (uint32_t offset = 1; offset < node_width; offset <<= 1) {
      auto up_exponent0 = tile_.shfl_up(exponent0, offset);
      if (tile_.thread_rank() >= offset) {
        exponent0 *= up_exponent0;
      }
    }
    // prime_multiplier = p^31
    static constexpr uint32_t prime0_multiplier = utils::constexpr_pow(prime0, node_max_len_);
    // 2. compute per-lane value
    auto this_length = get_key_length();
    uint32_t hash = 0;
    // ignore first one element in head;
    //  also make exponent [p^30, 1, p, p^2, ..., p^29, x]
    {
      auto shifted_exponent = tile_.shfl_down(exponent0, node_max_len_);
      exponent0 = tile_.shfl_up(exponent0, 1);
      if (tile_.thread_rank() < 1) { exponent0 = shifted_exponent; }
    }
    int skip_elems = 1;
    this_length += skip_elems;
    auto elem = lane_elem_;
    while (true) {
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < node_max_len_ &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem;
      }
      if (this_length <= node_max_len_) { break; }
      this_length -= node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      if (skip_elems <= tile_.thread_rank()) {
        exponent0 *= prime0_multiplier;
      }
      skip_elems = 0;
    }
    // 3. reduce sum
    for (uint32_t offset = (node_width / 2); offset != 0; offset >>= 1) {
      hash += tile_.shfl_down(hash, offset);
    }
    return tile_.shfl(hash, 0);
  }
  template <uint32_t prime0, uint32_t prime1>
  DEVICE_QUALIFIER uint2 compute_polynomialx2() const {
    // compute (1 * s[0]) + (p * s[1]) + (p^2 * s[2]) + ... + (p^(l-1) * s[l-1])
    // 1. exponent = [1, p, p^2, ..., p^31]
    uint32_t exponent0 = (tile_.thread_rank() == 0) ? 1 : prime0;
    uint32_t exponent1 = (tile_.thread_rank() == 0) ? 1 : prime1;
    for (uint32_t offset = 1; offset < node_width; offset <<= 1) {
      auto up_exponent0 = tile_.shfl_up(exponent0, offset);
      auto up_exponent1 = tile_.shfl_up(exponent1, offset);
      if (tile_.thread_rank() >= offset) {
        exponent0 *= up_exponent0;
        exponent1 *= up_exponent1;
      }
    }
    // prime_multiplier = p^31
    static constexpr uint32_t prime0_multiplier = utils::constexpr_pow(prime0, node_max_len_);
    static constexpr uint32_t prime1_multiplier = utils::constexpr_pow(prime1, node_max_len_);
    // 2. compute per-lane value
    auto this_length = get_key_length();
    uint32_t hash = 0, hash1 = 0;
    // ignore first one element in head;
    //  also make exponent [p^30, 1, p, p^2, ..., p^29, x]
    {
      auto shifted_exponent = tile_.shfl_down(exponent0, node_max_len_);
      exponent0 = tile_.shfl_up(exponent0, 1);
      if (tile_.thread_rank() < 1) { exponent0 = shifted_exponent; }
      shifted_exponent = tile_.shfl_down(exponent1, node_max_len_);
      exponent1 = tile_.shfl_up(exponent1, 1);
      if (tile_.thread_rank() < 1) { exponent1 = shifted_exponent; }
    }
    int skip_elems = 1;
    this_length += skip_elems;
    auto elem = lane_elem_;
    while (true) {
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < node_max_len_ &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem;
        hash1 += exponent1 * elem;
      }
      if (this_length <= node_max_len_) { break; }
      this_length -= node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      if (skip_elems <= tile_.thread_rank()) {
        exponent0 *= prime0_multiplier;
        exponent1 *= prime1_multiplier;
      }
      skip_elems = 0;
    }
    // 3. reduce sum
    for (uint32_t offset = (node_width / 2); offset != 0; offset >>= 1) {
      hash += tile_.shfl_down(hash, offset);
      hash1 += tile_.shfl_down(hash1, offset);
    }
    return make_uint2(tile_.shfl(hash, 0), tile_.shfl(hash1, 0));
  }

  template <typename keyptr_or_keystore>
  DEVICE_QUALIFIER void create_from(keyptr_or_keystore key, size_type key_length, const elem_type* value, size_type value_length) {
    // head node metadata
    elem_type elem;
    elem_type* curr_ptr = nullptr;  // NULL if head, else appendix
    if (tile_.thread_rank() == head_node_length_lane_) {
      elem = (key_length << key_length_offset_bits_) |
             (value_length << value_length_offset_bits_);
    }
    // ignore first one element in head
    int skip_elems = 1;
    int key_offset = -skip_elems;
    key_length += skip_elems;
    value -= key_length;
    value_length += key_length;
    while (true) {
      // set elem
      if (skip_elems <= tile_.thread_rank()) {
        if (tile_.thread_rank() < min(key_length, node_max_len_)) {
          elem = key[key_offset + tile_.thread_rank()];
        }
        else if (tile_.thread_rank() < min(value_length, node_max_len_)) {
          elem = value[tile_.thread_rank()];
        }
      }
      elem_type* next_ptr;
      if (value_length > node_max_len_) {
        auto next_index = allocator_.allocate(tile_);
        if (tile_.thread_rank() == next_lane_) { elem = next_index; }
        next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      }
      // store
      if (curr_ptr) { // !is_head
        utils::memory::store<elem_type, false>(curr_ptr + tile_.thread_rank(), elem);
      }
      else {  // is_head
        lane_elem_ = elem;
      }
      // proceed
      if (value_length <= node_max_len_) { break; }
      curr_ptr = next_ptr;
      key_length = max(static_cast<int>(key_length - node_max_len_), 0);
      key_offset += node_max_len_;
      value_length -= node_max_len_;
      value += node_max_len_;
      skip_elems = 0;
    }
  }

  DEVICE_QUALIFIER void flush(elem_type* key_buffer) {
    auto this_length = get_key_length();
    auto elem = lane_elem_;
    // ignore first one element in head
    int skip_elems = 1;
    int key_offset = -skip_elems;
    this_length += skip_elems;
    while (true) {
      auto count = min(this_length, node_max_len_);
      if (skip_elems <= tile_.thread_rank() && tile_.thread_rank() < count) {
        key_buffer[key_offset + tile_.thread_rank()] = elem;
      }
      this_length -= count;
      if (this_length == 0) { break; }
      key_offset += count;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void move_from(suffix_node_warp<tile_type, allocator_type>& src,
                                  uint32_t offset,
                                  reclaimer_type& reclaimer) {
    // move elements from src[offset:] and retire all nodes in src
    auto new_length = src.get_key_length() - offset;
    // skip src nodes until the first element
    reclaimer.retire(src.get_node_index(), tile_);
    while (offset >= node_max_len_) {
      src.node_index_ = src.get_next();
      src.load_head();
      reclaimer.retire(src.node_index_, tile_);
      offset -= node_max_len_;
    }
    // copy elements into this
    elem_type dst_lane_elem;
    elem_type* dst_ptr = nullptr; // NULL means it's head
    auto value_length = src.get_value_length();
    if (tile_.thread_rank() == head_node_length_lane_) {
      dst_lane_elem = (new_length << key_length_offset_bits_) |
                      (value_length << value_length_offset_bits_);
    }
    // ignore first one element in head, but also copy value
    int skip_elems = 1;
    new_length += (skip_elems + value_length);
    while (true) {
      // phase 1. copy src[offset:node_max_len) -> dst[0:node_max_len-offset)
      uint32_t copy_count = min(new_length, node_max_len_ - offset);
      auto down_elem = tile_.shfl_down(src.lane_elem_, offset);
      dst_lane_elem = (tile_.thread_rank() >= skip_elems) ? down_elem : dst_lane_elem;
      new_length -= copy_count;
      if (new_length == 0) { break; }
      // phase 2. copy src.next[0:offset) -> dst[node_max_len-offset:node_max_len)
      src.node_index_ = src.get_next();
      src.load_head();
      reclaimer.retire(src.node_index_, tile_);
      if (offset > 0) {
        copy_count = min(new_length, offset);
        auto up_src_elem = tile_.shfl_up(src.lane_elem_, node_max_len_ - offset);
        if (node_max_len_ - offset <= tile_.thread_rank() &&
            skip_elems <= tile_.thread_rank()) {
          dst_lane_elem = up_src_elem;
        }
        new_length -= copy_count;
        if (new_length == 0) { break; }
      }
      skip_elems = 0;
      // phase 3. store dst & allocate dst.next
      auto dst_index = allocator_.allocate(tile_);
      if (tile_.thread_rank() == next_lane_) {
        dst_lane_elem = dst_index;
      }
      if (dst_ptr) {
        utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
      }
      else {
        lane_elem_ = dst_lane_elem;
      }
      dst_ptr = reinterpret_cast<elem_type*>(allocator_.address(dst_index));
    }
    // flush dst_lane_elem
    if (dst_ptr) {
      utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
    }
    else {
      lane_elem_ = dst_lane_elem;
    }
  }

  DEVICE_QUALIFIER elem_type get_value() const {
    // return first value slice
    auto elem = lane_elem_;
    // ignore first (key_length + 1) element in head
    int skip_elems = get_key_length() + 1;
    while (true) {
      if (skip_elems < node_max_len_) {
        return tile_.shfl(elem, skip_elems);
      }
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      skip_elems = max(static_cast<int>(skip_elems - node_max_len_), 0);
    }
  }

  DEVICE_QUALIFIER void get_value(elem_type* value_buffer, size_type max_value_length) const {
    auto value_length = min(get_value_length(), max_value_length);
    auto elem = lane_elem_;
    // ignore first (key_length + 1) element in head
    int skip_elems = get_key_length() + 1;
    int value_offset = -skip_elems;
    value_length += skip_elems;
    while (true) {
      auto count = min(value_length, node_max_len_);
      if (skip_elems <= tile_.thread_rank() && tile_.thread_rank() < count) {
        value_buffer[value_offset + tile_.thread_rank()] = elem;
      }
      value_length -= count;
      if (value_length == 0) { break; }
      value_offset += count;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      skip_elems = max(static_cast<int>(skip_elems - node_max_len_), 0);
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void switch_value_from(suffix_node_warp<tile_type, allocator_type>& src,
                                          const elem_type* value,
                                          size_type value_length,
                                          reclaimer_type& reclaimer) {
    // move elements from src, but switch to new value
    auto key_length = src.get_key_length();
    elem_type dst_lane_elem;
    elem_type* dst_ptr = nullptr; // NULL means it's head
    if (tile_.thread_rank() == head_node_length_lane_) {
      dst_lane_elem = (key_length << key_length_offset_bits_) |
                      (value_length << value_length_offset_bits_);
    }
    // ignore first one element in head
    auto src_nodes_left = src.get_num_nodes() - 1;
    reclaimer.retire(src.get_node_index(), tile_);
    int skip_elems = 1;
    key_length += skip_elems;
    auto total_length = key_length + value_length;
    int value_offset = -static_cast<int>(key_length);
    while (true) {
      if (skip_elems <= tile_.thread_rank()) {
        if (tile_.thread_rank() < min(key_length, node_max_len_)) {
          dst_lane_elem = src.lane_elem_;
        }
        else if (tile_.thread_rank() < min(total_length, node_max_len_)) {
          dst_lane_elem = value[value_offset + tile_.thread_rank()];
        }
      }
      total_length = max(static_cast<int>(total_length - node_max_len_), 0);
      if (total_length == 0) { break; }
      // store dst & allocate dst.next
      auto dst_index = allocator_.allocate(tile_);
      if (tile_.thread_rank() == next_lane_) {
        dst_lane_elem = dst_index;
      }
      if (dst_ptr) {
        utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
      }
      else {
        lane_elem_ = dst_lane_elem;
      }
      dst_ptr = reinterpret_cast<elem_type*>(allocator_.address(dst_index));
      if (src_nodes_left > 0) {
        src.node_index_ = src.get_next();
        src.load_head();
        reclaimer.retire(src.node_index_, tile_);
        src_nodes_left--;
      }
      value_offset += node_max_len_;
      key_length = max(static_cast<int>(key_length - node_max_len_), 0);
      skip_elems = 0;
    }
    // flush dst_lane_elem
    if (dst_ptr) {
      utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
    }
    else {
      lane_elem_ = dst_lane_elem;
    }
    while (src_nodes_left > 0) {
      src.node_index_ = src.get_next();
      src.load_head();
      reclaimer.retire(src.node_index_, tile_);
      src_nodes_left--;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void retire(reclaimer_type& reclaimer) {
    reclaimer.retire(node_index_, tile_);
    auto num_nodes = get_num_nodes() - 1;
    auto suffix_index = get_next();
    while (num_nodes > 0) {
      auto* suffix_ptr = reinterpret_cast<elem_type*>(allocator_.address(suffix_index));
      elem_type next_index;
      if (tile_.thread_rank() == 0) {
        next_index = utils::memory::load<elem_type, false>(suffix_ptr + next_lane_);
      }
      next_index = tile_.shfl(next_index, 0);
      reclaimer.retire(suffix_index, tile_);
      suffix_index = next_index;
      num_nodes--;
    }
  }

  DEVICE_QUALIFIER suffix_node_warp<tile_type, allocator_type>& operator=(
      const suffix_node_warp<tile_type, allocator_type>& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    auto length = get_key_length();
    auto value_length = get_value_length();
    if (lead_lane) printf("node[%u]: s{kl=%u vl=%u; K= ", node_index_, length, value_length);
    length += 1;
    auto total_length = length + value_length;
    for (uint32_t i = 1; i < min(total_length, node_max_len_); i++) {
      if (i == length && lead_lane) printf("; V= ");
      elem_type key_slice = tile_.shfl(lane_elem_, i);
      if (lead_lane) printf("%u ", key_slice);
    }
    if (total_length <= node_max_len_) {
      if (lead_lane) printf("}\n");
    }
    else {
      auto next = get_next();
      if (lead_lane) printf("(n=%u)}\n", next);
      total_length -= node_max_len_;
      length -= min(node_max_len_, length);
      while (true) {
        if (lead_lane) printf("node[%u]: s.{", next);
        auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next));
        auto elem = next_ptr[tile_.thread_rank()];
        for (uint32_t i = 0; i < min(total_length, node_max_len_); i++) {
          elem_type key_slice = tile_.shfl(elem, i);
          if (i == length && lead_lane) printf("; V= ");
          if (lead_lane) printf("%u ", key_slice);
        }
        if (total_length <= node_max_len_) {
          if (lead_lane) printf("}\n");
          break;
        }
        else {
          next = tile_.shfl(elem, next_lane_);
          if (lead_lane) printf("(n=%u)}\n", next);
          total_length -= node_max_len_;
          length -= min(node_max_len_, length);
        }
      }
    }
  }

 private:
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;

  // each node consists of 32 elements and stores up to 31 key slices
  // [slice0] [slice1] ... [slice28] [slice29] [slice30] [next]
  // At the head node, slice0 = (value_length: 16 | key_length: 16), value comes after key

  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t head_node_length_lane_ = 0;
  static constexpr uint32_t next_lane_ = node_width - 1;
  static constexpr uint32_t node_max_len_ = node_width - 1;
  static constexpr uint32_t length_mask_ = (1u << 16) - 1;
  static constexpr uint32_t key_length_offset_bits_ = 0;
  static constexpr uint32_t value_length_offset_bits_ = 16;
};
