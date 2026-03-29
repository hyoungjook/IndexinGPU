// for compatibility with ARTSynchronized
#pragma once

namespace tbb {

template <typename T>
class combinable {
 public:
  T& local() {
    thread_local T value{};
    return value;
  }
  const T& local() const {
    return const_cast<combinable*>(this)->local();
  }
};

}  // namespace tbb
