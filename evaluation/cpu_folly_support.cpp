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

#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <typeindex>
#include <unordered_map>

#include <folly/Function.h>
#include <folly/ScopeGuard.h>
#include <folly/detail/StaticSingletonManager.h>
#include <folly/synchronization/Hazptr.h>

namespace folly {
namespace detail {

void ScopeGuardImplBase::terminate() noexcept {
  std::terminate();
}

namespace {

class StaticSingletonManagerWithRttiImpl {
 public:
  using Make = void*();

  struct Entry {
    std::atomic<void*> ptr{nullptr};
    std::mutex mutex;

    void* get_existing() const {
      return ptr.load(std::memory_order_acquire);
    }

    void* create(Make& make, void*& debug) {
      if (auto* value = ptr.load(std::memory_order_acquire)) {
        return value;
      }
      std::unique_lock<std::mutex> lock(mutex);
      if (auto* value = ptr.load(std::memory_order_acquire)) {
        return value;
      }
      auto* value = make();
      ptr.store(value, std::memory_order_release);
      debug = value;
      return value;
    }
  };

  static StaticSingletonManagerWithRttiImpl& instance() {
    static auto* manager = new StaticSingletonManagerWithRttiImpl();
    return *manager;
  }

  Entry* get_existing_entry(const std::type_info& key) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = map_.find(std::type_index(key));
    return iter == map_.end() ? nullptr : &iter->second;
  }

  Entry& create_entry(const std::type_info& key) {
    std::unique_lock<std::mutex> lock(mutex_);
    return map_[std::type_index(key)];
  }

 private:
  std::unordered_map<std::type_index, Entry> map_;
  std::mutex mutex_;
};

} // namespace

void* StaticSingletonManagerWithRtti::get_existing_(Arg& arg) noexcept {
  auto* entry = StaticSingletonManagerWithRttiImpl::instance().get_existing_entry(*arg.key);
  auto* value = entry ? entry->get_existing() : nullptr;
  if (value) {
    arg.cache.store(value, std::memory_order_release);
  }
  return value;
}

template <bool Noexcept>
void* StaticSingletonManagerWithRtti::create_(Arg& arg) noexcept(Noexcept) {
  auto& entry = StaticSingletonManagerWithRttiImpl::instance().create_entry(*arg.key);
  auto* value = entry.create(*arg.make, *arg.debug);
  arg.cache.store(value, std::memory_order_release);
  return value;
}

template void* StaticSingletonManagerWithRtti::create_<false>(Arg& arg);
template void* StaticSingletonManagerWithRtti::create_<true>(Arg& arg) noexcept;

void hazptr_inline_executor_add(folly::Function<void()> func) {
  func();
}

} // namespace detail

FOLLY_STATIC_CTOR_PRIORITY_MAX hazptr_domain<std::atomic> default_domain;

bool hazptr_use_executor() {
  return false;
}

} // namespace folly
