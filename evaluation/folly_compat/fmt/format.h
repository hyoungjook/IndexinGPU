#pragma once

#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>

#define FMT_VERSION 100000

namespace fmt {

template <typename Char>
class basic_string_view {
 public:
  using value_type = Char;

  constexpr basic_string_view() noexcept : data_(nullptr), size_(0) {}

  constexpr basic_string_view(const Char* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  constexpr basic_string_view(std::basic_string_view<Char> view) noexcept
      : data_(view.data()), size_(view.size()) {}

  constexpr const Char* data() const noexcept {
    return data_;
  }

  constexpr std::size_t size() const noexcept {
    return size_;
  }

 private:
  const Char* data_;
  std::size_t size_;
};

using string_view = basic_string_view<char>;

struct format_parse_context {
  using iterator = const char*;

  constexpr iterator begin() const noexcept {
    return nullptr;
  }
};

template <typename OutputIt, typename Char = char>
struct basic_format_context {
  using iterator = OutputIt;

  iterator out() {
    return iterator{};
  }
};

using format_context = basic_format_context<std::back_insert_iterator<std::string>>;

template <typename T, typename Char = char, typename Enable = void>
struct formatter {
  constexpr typename format_parse_context::iterator parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  typename FormatContext::iterator format(const T&, FormatContext& ctx) const {
    return ctx.out();
  }
};

template <typename... Args>
using format_string = string_view;

template <typename... Args>
inline std::string format(format_string<Args...>, Args&&...) {
  return {};
}

} // namespace fmt
