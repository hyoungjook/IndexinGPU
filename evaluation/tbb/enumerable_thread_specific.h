// for compatibility with ARTSynchronized
#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tbb {

template <typename T>
class enumerable_thread_specific {
 public:
  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    explicit iterator(T** current) : current_(current) {
    }

    reference operator*() const { return **current_; }
    pointer operator->() const { return *current_; }
    iterator& operator++() {
      ++current_;
      return *this;
    }
    bool operator==(const iterator& other) const { return current_ == other.current_; }
    bool operator!=(const iterator& other) const { return !(*this == other); }

   private:
    T** current_;
  };

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    explicit const_iterator(T* const* current) : current_(current) {
    }

    reference operator*() const { return **current_; }
    pointer operator->() const { return *current_; }
    const_iterator& operator++() {
      ++current_;
      return *this;
    }
    bool operator==(const const_iterator& other) const { return current_ == other.current_; }
    bool operator!=(const const_iterator& other) const { return !(*this == other); }

   private:
    T* const* current_;
  };

  T& local() {
    const auto thread_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = locals_.find(thread_id);
    if (it != locals_.end()) {
      return *it->second;
    }
    storage_.push_back(std::make_unique<T>());
    T* value = storage_.back().get();
    locals_.emplace(thread_id, value);
    return *value;
  }

  iterator begin() {
    refresh_snapshot();
    return iterator(iteration_snapshot_.data());
  }
  iterator end() {
    return iterator(iteration_snapshot_.data() + iteration_snapshot_.size());
  }
  const_iterator begin() const {
    refresh_snapshot();
    return const_iterator(iteration_snapshot_.data());
  }
  const_iterator end() const {
    return const_iterator(iteration_snapshot_.data() + iteration_snapshot_.size());
  }

 private:
  void refresh_snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    iteration_snapshot_.clear();
    iteration_snapshot_.reserve(storage_.size());
    for (const auto& value : storage_) {
      iteration_snapshot_.push_back(value.get());
    }
  }

  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<T>> storage_;
  std::unordered_map<std::thread::id, T*> locals_;

  inline static thread_local std::vector<T*> iteration_snapshot_;
};

}  // namespace tbb
