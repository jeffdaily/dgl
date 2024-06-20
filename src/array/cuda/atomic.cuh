/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cuda/atomic.cuh
 * @brief Atomic functions
 */
#ifndef DGL_ARRAY_CUDA_ATOMIC_CUH_
#define DGL_ARRAY_CUDA_ATOMIC_CUH_

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "bf16.cuh"
#include "fp16.cuh"

#if __CUDA_ARCH__ >= 600 || defined(DGL_USE_ROCM)
#include <cuda_fp16.h>
#endif

namespace dgl {
namespace aten {
namespace cuda {

// Type trait for selecting code type
template <int Bytes>
struct Code {};

template <>
struct Code<2> {
  typedef unsigned short int Type;  // NOLINT
};

template <>
struct Code<4> {
  typedef unsigned int Type;  // NOLINT
};

template <>
struct Code<8> {
  typedef unsigned long long int Type;  // NOLINT
};

// Helper class for converting to/from atomicCAS compatible types.
template <typename T>
struct Cast {
  typedef typename Code<sizeof(T)>::Type Type;
  static __device__ __forceinline__ Type Encode(T val) {
    return static_cast<Type>(val);
  }
  static __device__ __forceinline__ T Decode(Type code) {
    return static_cast<T>(code);
  }
};

template <>
struct Cast<half> {
  typedef Code<sizeof(half)>::Type Type;
  static __device__ __forceinline__ Type Encode(half val) {
    return __half_as_ushort(val);
  }
  static __device__ __forceinline__ half Decode(Type code) {
    return __ushort_as_half(code);
  }
};

#if BF16_ENABLED
template <>
struct Cast<__nv_bfloat16> {
  typedef Code<sizeof(__nv_bfloat16)>::Type Type;
  static __device__ __forceinline__ Type Encode(__nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat16_as_ushort(val);
#else
    printf(
        "Atomic operations are not supported for bfloat16 (BF16) "
        "on GPUs with compute capability less than 8.0.\n");
    __trap();
    return static_cast<Type>(0);
#endif
  }
  static __device__ __forceinline__ __nv_bfloat16 Decode(Type code) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __ushort_as_bfloat16(code);
#else
    printf(
        "Atomic operations are not supported for bfloat16 (BF16) "
        "on GPUs with compute capability less than 8.0.\n");
    __trap();
    return static_cast<__nv_bfloat16>(0.0f);
#endif
  }
};
#endif  // BF16_ENABLED

template <>
struct Cast<float> {
  typedef Code<sizeof(float)>::Type Type;
  static __device__ __forceinline__ Type Encode(float val) {
    return __float_as_uint(val);
  }
  static __device__ __forceinline__ float Decode(Type code) {
    return __uint_as_float(code);
  }
};

template <>
struct Cast<double> {
  typedef Code<sizeof(double)>::Type Type;
  static __device__ __forceinline__ Type Encode(double val) {
    return __double_as_longlong(val);
  }
  static __device__ __forceinline__ double Decode(Type code) {
    return __longlong_as_double(code);
  }
};

#ifndef DGL_USE_ROCM
static __device__ __forceinline__ unsigned short int atomicCASshort(  // NOLINT
    unsigned short int* address,                                      // NOLINT
    unsigned short int compare,                                       // NOLINT
    unsigned short int val) {                                         // NOLINT
  static_assert(CUDART_VERSION >= 10000, "Requires at least CUDA 10");
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 700)
  return atomicCAS(address, compare, val);
#else
  (void)address;
  (void)compare;
  (void)val;
  printf(
      "Atomic operations are not supported for half precision (FP16) "
      "on this GPU.\n");
  __trap();
  return val;
#endif  // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__) >= 700)
}
#else // DGL_USE_ROCM
template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<half> {
  template <typename func_t>
  inline __device__ half operator() (half *address, half val, const func_t& func) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    __half_raw hsum;
    do {
      assumed = old;
      hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      hsum = func(hsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return hsum;
  }
};

template <typename T>
inline __host__ __device__ T safe_max(T a, T b) {
  #if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
    T max = ::isnan(a) ? a : (::isnan(b) ? b : std::max<T>(a, b));
  #else
    T max = ::isnan(b) ? b : std::max<T>(a, b);
  #endif

  return max;
}

template <>
inline __host__ __device__ half safe_max<half>(half a, half b) {
  #if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
    half max = ::__hisnan(a) ? a : (::__hisnan(b) ? b : std::max<half>(a, b));
  #else
    half max = ::__hisnan(b) ? b : std::max<half>(a, b);
  #endif

  return max;
}

template <typename T>
inline __host__ __device__ T safe_min(T a, T b) {
  #if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
    T min = ::isnan(a) ? a : (::isnan(b) ? b : std::min<T>(a, b));
  #else
    T min = ::isnan(b) ? b : std::min<T>(a, b);
  #endif

  return min;
}

template <>
inline __host__ __device__ half safe_min(half a, half b) {
  #if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
    half min = ::__hisnan(a) ? a : (::__hisnan(b) ? b : std::min<half>(a, b));
  #else
    half min = ::__hisnan(b) ? b : std::min<half>(a, b);
  #endif

  return min;
}
#endif

#define DEFINE_ATOMIC(NAME)                                   \
  template <typename T>                                       \
  __device__ __forceinline__ T Atomic##NAME(T* addr, T val) { \
    typedef typename Cast<T>::Type CT;                        \
    CT* addr_as_ui = reinterpret_cast<CT*>(addr);             \
    CT old = *addr_as_ui;                                     \
    CT assumed = old;                                         \
    do {                                                      \
      assumed = old;                                          \
      old = atomicCAS(                                        \
          addr_as_ui, assumed,                                \
          Cast<T>::Encode(OP(val, Cast<T>::Decode(old))));    \
    } while (assumed != old);                                 \
    return Cast<T>::Decode(old);                              \
  }

#ifndef DGL_USE_ROCM
#define DEFINE_ATOMIC_16BIT(NAME, dtype)                           \
  template <>                                                      \
  __device__ __forceinline__ dtype Atomic##NAME<dtype>(            \
      dtype * addr, dtype val) {                                   \
    typedef uint16_t CT;                                           \
    CT* addr_as_ui = reinterpret_cast<CT*>(addr);                  \
    CT old = *addr_as_ui;                                          \
    CT assumed = old;                                              \
    do {                                                           \
      assumed = old;                                               \
      old = atomicCASshort(                                        \
          addr_as_ui, assumed,                                     \
          Cast<dtype>::Encode(OP(val, Cast<dtype>::Decode(old)))); \
    } while (assumed != old);                                      \
    return Cast<dtype>::Decode(old);                               \
  }
#else // DGL_USE_ROCM
#define DEFINE_ATOMIC_16BIT(NAME, dtype)                           \
  inline __device__ dtype Atomic##NAME(dtype * addr, dtype val) {  \
    return AtomicFPOp<dtype>()(addr, val, [](dtype a, dtype b) { return OP_; }); \
  }
#endif

#ifdef DGL_USE_ROCM
#define OP(a, b) safe_max(a, b)
#define OP_      safe_max(a, b)
#else
#define OP(a, b) max(a, b)
#endif
DEFINE_ATOMIC(Max)
DEFINE_ATOMIC_16BIT(Max, half)
#if BF16_ENABLED
DEFINE_ATOMIC_16BIT(Max, __nv_bfloat16)
#endif  // BF16_ENABLED
#undef OP
#undef OP_

#ifdef DGL_USE_ROCM
#define OP(a, b) safe_min(a, b)
#define OP_      safe_min(a, b)
#else
#define OP(a, b) min(a, b)
#endif
DEFINE_ATOMIC(Min)
DEFINE_ATOMIC_16BIT(Min, half)
#if BF16_ENABLED
DEFINE_ATOMIC_16BIT(Min, __nv_bfloat16)
#endif  // BF16_ENABLED
#undef OP
#undef OP_

#define OP(a, b) a + b
DEFINE_ATOMIC(Add)
#undef OP

/**
 * @brief Performs an atomic compare-and-swap on 64 bit integers. That is,
 * it the word `old` at the memory location `address`, computes
 * `(old == compare ? val : old)` , and stores the result back to memory at
 * the same address.
 *
 * @param address The address to perform the atomic operation on.
 * @param compare The value to compare to.
 * @param val The new value to conditionally store.
 *
 * @return The old value at the address.
 */
inline __device__ int64_t
AtomicCAS(int64_t* const address, const int64_t compare, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

/**
 * @brief Performs an atomic compare-and-swap on 32 bit integers. That is,
 * it the word `old` at the memory location `address`, computes
 * `(old == compare ? val : old)` , and stores the result back to memory at
 * the same address.
 *
 * @param address The address to perform the atomic operation on.
 * @param compare The value to compare to.
 * @param val The new value to conditionally store.
 *
 * @return The old value at the address.
 */
inline __device__ int32_t
AtomicCAS(int32_t* const address, const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

inline __device__ int64_t AtomicMax(int64_t* const address, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicMax(int32_t* const address, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address), static_cast<Type>(val));
}

template <>
__device__ __forceinline__ float AtomicAdd<float>(float* addr, float val) {
#if __CUDA_ARCH__ >= 200 || defined(DGL_USE_ROCM)
  return atomicAdd(addr, val);
#else
  typedef float T;
  typedef typename Cast<T>::Type CT;
  CT* addr_as_ui = reinterpret_cast<CT*>(addr);
  CT old = *addr_as_ui;
  CT assumed = old;
  do {
    assumed = old;
    old = atomicCAS(
        addr_as_ui, assumed, Cast<T>::Encode(Cast<T>::Decode(old) + val));
  } while (assumed != old);
  return Cast<T>::Decode(old);
#endif  // __CUDA_ARCH__ || defined(DGL_USE_ROCM)
}

template <>
__device__ __forceinline__ double AtomicAdd<double>(double* addr, double val) {
#if __CUDA_ARCH__ >= 600 || defined(DGL_USE_ROCM)
  return atomicAdd(addr, val);
#else
  typedef double T;
  typedef typename Cast<T>::Type CT;
  CT* addr_as_ui = reinterpret_cast<CT*>(addr);
  CT old = *addr_as_ui;
  CT assumed = old;
  do {
    assumed = old;
    old = atomicCAS(
        addr_as_ui, assumed, Cast<T>::Encode(Cast<T>::Decode(old) + val));
  } while (assumed != old);
  return Cast<T>::Decode(old);
#endif
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000 || defined(DGL_USE_ROCM)
template <>
__device__ __forceinline__ half AtomicAdd<half>(half* addr, half val) {
// make sure we have half support
#if __CUDA_ARCH__ >= 700
  return atomicAdd(addr, val);
#elif defined(DGL_USE_ROCM)
  return AtomicFPOp<half>()(addr, val, [](half hsum, half val) { return hsum + val; });
#else
  (void)addr;
  (void)val;
  printf(
      "Atomic operations are not supported for half precision (FP16) "
      "on this GPU.\n");
  __trap();
  return val;
#endif  // __CUDA_ARCH__ >= 700
}
#endif  // defined(CUDART_VERSION) && CUDART_VERSION >= 10000

#if BF16_ENABLED
template <>
__device__ __forceinline__ __nv_bfloat16
AtomicAdd<__nv_bfloat16>(__nv_bfloat16* addr, __nv_bfloat16 val) {
// make sure we have bfloat16 support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return atomicAdd(addr, val);
#else
  (void)addr;
  (void)val;
  printf(
      "Atomic operations are not supported for bfloat16 (BF16) "
      "on GPUs with compute capability less than 8.0.\n");
  __trap();
  return val;
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
}
#endif  // BF16_ENABLED

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_ATOMIC_CUH_
