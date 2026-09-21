// Portability shim for older toolchains.
//
// MinGW.org GCC 6.3's libstdc++ predates std::clamp (added in GCC 7) even
// though it accepts -std=c++17. Including this header first supplies it.
// On any compiler that already has std::clamp this file does nothing.
//
// It also pulls in <cmath> with _USE_MATH_DEFINES so that M_PI is visible:
// MinGW hides M_PI under strict ISO mode (-std=c++17 rather than -std=gnu++17).
#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cmath>

#if !defined(__cpp_lib_clamp)
namespace std {
template <class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
  return v < lo ? lo : (hi < v ? hi : v);
}
template <class T, class Compare>
constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp) {
  return comp(v, lo) ? lo : (comp(hi, v) ? hi : v);
}
}  // namespace std
#endif

// Some MinGW headers still don't define M_PI even with _USE_MATH_DEFINES
// when strict ISO mode is on; define it as a last resort.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
