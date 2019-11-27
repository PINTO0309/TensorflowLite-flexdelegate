/* Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// # What is "packing"?
//
// Before feeding data to the gemm kernels (the parts of Ruy that do lots
// of multiply-add operations), Ruy first performs a data transformation (which
// we call "packing") on the input matrices. This transformation has two main
// goals:
// - rearrange data into blocks that are a convenient size/layout for the gemm
// kernels to consume. This helps make the memory access pattern of the gemm
// kernel simpler and more contiguous, and puts the data in a layout most
// convenient for specific arithmetic instructions in the gemm kernel.
// - compute row/column sums needed for handling quantization with non-symmetric
// zero points.
//
// # Simplified algorithmic analysis of packing
//
// Packing is a relatively simple transformation which does a small constant
// amount of work on each element of an input matrix, and hence for an NxM
// matrix performs O(N*M) work. If N and M are of the same order, then this is
// O(N^2) work.
//
// A NxKxM matrix multiplication requires N*K*M multiply-accumulate operations.
// Note that if N, K, and M are all the same order, then the number of
// multiply-accumulate operations is O(N^3).
//
// Thus, the O(N^2) cost of packing is small compared to the O(N^3) work, in the
// case of all dimensions being roughly the same order.
//
// # Packing cost can be significant
//
// When matrix * matrix multiplications begin to look more like matrix * vector
// multiplications, packing cost can become significant. We sometimes call these
// cases "gemv-like".
//
// Continuing the algorithmic analysis above, if we consider a case where an
// NxKxM matrix multiplication has either N = O(1) or M = O(1), then the
// situation is different. In this case, the multiply-accumulate work is only
// quadratic, so the quadratic cost of packing can be come significant.
//
// Another way to say this is that the cost of packing an input matrix (either
// the LHS or RHS) is amortized across the non-depth dimension of the opposite
// input matrix. Thus, when the LHS has very few rows or the RHS has very few
// columns, the cost of packing the opposite input matrix can become
// significant.
//
// As a rough rule of thumb, the cost of packing starts to become significant
// when either N or M is below 32 (and other dimensions are hundreds), with very
// significant packing costs at 8 or below. This varies by data type, Path, and
// tuning, so these numbers are only rough guides.
//
// One practical use case that is affected by this is inference of
// fully connected neural network layers with a low batch size. The weight
// matrix (which is a constant for inference) is the one affected by significant
// packing cost.
//
// Ruy provides an API in ruy_advanced.h for advanced users to pre-pack
// input matrices that are affected by significant packing costs.
//
// # Implementation notes
//
// Ruy's packing routines always operate on a range of columns and can be
// applied to either the LHS or RHS. This is possible because Ruy internally
// implements a TrMul, so the accumulation along depth is done along columns of
// both the LHS and RHS (whereas for a normal Mul the accumulation along depth
// for the LHS is along rows). As another example, we are always computing
// column sums for quantization (and never row sums, since the LHS is
// transposed).

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_

#include <cstdint>
#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

template <Path ThePath, typename Scalar>
struct PackedTypeImpl {
  using Type = Scalar;
};

#if RUY_PLATFORM(NEON)
template <>
struct PackedTypeImpl<Path::kNeon, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kNeonDotprod, std::uint8_t> {
  using Type = std::int8_t;
};
#elif RUY_PLATFORM(AVX512)
template <>
struct PackedTypeImpl<Path::kAvx512, std::uint8_t> {
  using Type = std::int8_t;
};
#endif

template <Path ThePath, typename Scalar>
using PackedType = typename PackedTypeImpl<ThePath, Scalar>::Type;

template <typename PackedScalar, typename Scalar>
PackedScalar Pack(Scalar x) {
  return x - SymmetricZeroPoint<Scalar>() + SymmetricZeroPoint<PackedScalar>();
}

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename SumsType>
struct PackImpl {};

#define RUY_INHERIT_PACK(PARENT, CHILD)                                       \
  template <typename FixedKernelLayout, typename Scalar,                      \
            typename PackedScalar, typename SumsType>                         \
  struct PackImpl<CHILD, FixedKernelLayout, Scalar, PackedScalar, SumsType>   \
      : PackImpl<PARENT, FixedKernelLayout, Scalar, PackedScalar, SumsType> { \
  };

template <typename FixedKernelLayout, typename Scalar, typename PackedScalar,
          typename SumsType>
struct PackImpl<Path::kStandardCpp, FixedKernelLayout, Scalar, PackedScalar,
                SumsType> {
  static void Run(Tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<PackedScalar>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (generic)");
    RUY_DCHECK_EQ((end_col - start_col) % FixedKernelLayout::kCols, 0);
    SumsType* sums = packed_matrix->sums;
    for (int col = start_col; col < end_col; col++) {
      SumsType accum = 0;
      for (int row = 0; row < packed_matrix->layout.rows; row++) {
        PackedScalar packed_val;
        if (col < src_matrix.layout.cols && row < src_matrix.layout.rows) {
          packed_val = Pack<PackedScalar>(Element(src_matrix, row, col));
        } else {
          packed_val = packed_matrix->zero_point;
        }
        accum += packed_val;
        *ElementPtr(packed_matrix, row, col) = packed_val;
      }
      if (sums) {
        sums[col] = accum;
      }
    }
  }
};

#if RUY_PLATFORM(NEON)
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kNeon)
#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
RUY_INHERIT_PACK(Path::kNeon, Path::kNeonDotprod)
#endif
#elif RUY_PLATFORM(AVX512)
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kAvx512)
#endif

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void Pack8bitNeonOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                            const void* src_ptr2, const void* src_ptr3,
                            int src_inc0, int src_inc1, int src_inc2,
                            int src_inc3, int src_rows, int src_zero_point,
                            std::int8_t* packed_ptr, int start_col, int end_col,
                            std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonInOrder(const void* src_ptr0, const void* src_ptr1,
                         const void* src_ptr2, const void* src_ptr3,
                         int src_inc0, int src_inc1, int src_inc2, int src_inc3,
                         int src_rows, int src_zero_point,
                         std::int8_t* packed_ptr, int start_col, int end_col,
                         std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                                   const void* src_ptr2, const void* src_ptr3,
                                   int src_inc0, int src_inc1, int src_inc2,
                                   int src_inc3, int src_rows,
                                   int src_zero_point, std::int8_t* packed_ptr,
                                   int start_col, int end_col,
                                   std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodInOrder(const void* src_ptr0, const void* src_ptr1,
                                const void* src_ptr2, const void* src_ptr3,
                                int src_inc0, int src_inc1, int src_inc2,
                                int src_inc3, int src_rows, int src_zero_point,
                                std::int8_t* packed_ptr, int start_col,
                                int end_col, std::int32_t* sums_ptr,
                                int input_xor);

template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 4>, Scalar,
                std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      int src_inc2 = 16;
      int src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
    }
  }
};

template <typename Scalar>
struct PackImpl<Path::kNeonDotprod, FixedKernelLayout<Order::kColMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & ~7) +
          ((block_col & 4) * 4);
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonDotprodInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonDotprodOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
    }
  }
};
#endif  // (RUY_PLATFORM(NEON_64)&& RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc0, int src_inc1, int src_inc2,
                             int src_inc3, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col);
void PackFloatNeonInOrder(const float* src_ptr0, const float* src_ptr1,
                          const float* src_ptr2, const float* src_ptr3,
                          int src_inc0, int src_inc1, int src_inc2,
                          int src_inc3, int src_rows, int src_zero_point,
                          float* packed_ptr, int start_col, int end_col);

#elif RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col,
                             int stride);
#endif  // (RUY_PLATFORM(NEON_64)&& RUY_OPT_ENABLED(RUY_OPT_ASM)

#if (RUY_PLATFORM(NEON_32) || RUY_PLATFORM(NEON_64)) && \
    RUY_OPT_ENABLED(RUY_OPT_ASM)

template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
                float, float> {
  static void Run(Tuning tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr = packed_matrix->data +
                          packed_matrix->layout.stride * (block_col & ~7) +
                          ((block_col & 4));
#if RUY_PLATFORM(NEON_64)
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        PackFloatNeonInOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0,
                             src_inc1, src_inc2, src_inc3,
                             src_matrix.layout.rows, src_matrix.zero_point,
                             packed_ptr, start_col, end_col);
      } else {
        PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                src_inc0, src_inc1, src_inc2, src_inc3,
                                src_matrix.layout.rows, src_matrix.zero_point,
                                packed_ptr, start_col, end_col);
      }
#else
      // Encode each of src_inc0, ..., src_inc3 in lowest 4 bits of src_inc
      // to save on registers (we have fewer general purpose registers in
      // 32-bit ARM than in 64-bit ARM). For the 64-bit case, we pass four
      // values that are each either 16 or 0 and use them directly. For the
      // 32-bit case, bits 0, 1, 2, and 3 are used to determine if we should
      // use the value 16 (bit is set) or 0 (bit is not set) for the
      // respective increment value.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 32;
      PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                              src_matrix.layout.rows, src_matrix.zero_point,
                              packed_ptr, start_col, end_col, kOutputStride);
#endif  // RUY_PLATFORM(NEON_64)
    }
  }
};

#if RUY_PLATFORM(NEON_32)
// The 32-bit float kernel is 8 rows X 4 columns, so we need an additional
// specialization for a FixedKernelLayout with 4 columns.
template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 4>, float,
                float, float> {
  static void Run(Tuning tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * (block_col);
      // Encode each of src_inc0, ..., src_inc1 in lowest 4 bits of scrc_inc
      // to save registers.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 16;
      PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                              src_matrix.layout.rows, src_matrix.zero_point,
                              packed_ptr, start_col, end_col, kOutputStride);
    }
  }
};
#endif  // (RUY_PLATFORM(NEON_32))
#endif  // (RUY_PLATFORM(NEON_64) || RUY_PLATFORM(NEON_32)) && \
        // RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)
// Note that source and zero buffers can be uint8 type, but in the packing
// function are reinterpreted as int8, and are XOR-ed with input_xor.
void Pack8bitAvx512(const std::int8_t* src_ptr, std::int8_t input_xor,
                    const std::int8_t* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows,
                    std::int8_t* packed_ptr, std::int32_t* sums_ptr);

template <typename Scalar>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kColMajor, 4, 16>,
                Scalar, std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  using Layout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  static constexpr int kHalfLayoutCols =
      8;  // Half the number of cols in a block.
  static constexpr std::int8_t kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (AVX-512)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    RUY_DCHECK_EQ(kHalfLayoutCols * 2, Layout::kCols);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[kHalfLayoutCols * Layout::kRows];
    memset(zerobuf, packed_matrix->zero_point ^ kInputXor,
           kHalfLayoutCols * Layout::kRows * sizeof(Scalar));
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      Pack8bitAvx512(reinterpret_cast<const std::int8_t*>(src_ptr), kInputXor,
                     reinterpret_cast<const std::int8_t*>(zerobuf), src_stride,
                     remaining_src_cols, src_matrix.layout.rows, packed_ptr,
                     sums_ptr);
    }
  }
};

void PackFloatAvx512(const float* src_ptr, const float* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kRowMajor, 1, 16>,
                float, float, float> {
  static void Run(Tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    using Layout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatAvx512(src_ptr, zerobuf, src_stride, remaining_src_cols,
                      src_matrix.layout.rows, packed_ptr);
    }
  }
};
#endif  // RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// Main entry point for packing.
template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar>
void RunPack(Tuning tuning, const DMatrix& src_matrix, PMatrix* packed_matrix,
             int start_col, int end_col) {
  using SumsType = typename PackedMatrix<PackedScalar>::SumsType;
  Matrix<Scalar> src = ToMatrix<Scalar>(src_matrix);
  PackedMatrix<PackedScalar> packed =
      ToPackedMatrix<PackedScalar>(*packed_matrix);
  PackImpl<ThePath, FixedKernelLayout, Scalar, PackedScalar, SumsType>::Run(
      tuning, src, &packed, start_col, end_col);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
