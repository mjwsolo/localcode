#pragma once

#include "ggml-metal-impl.h"

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

static inline float dot(float x, float y) {
    return x*y;
}

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
inline T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}

// ===== TurboQuant shared tables and helpers (fork-local) =====
// ----- TurboQuant quantize/dequantize with Fast Walsh-Hadamard rotation -----
// Uses O(d log d) WHT instead of O(d²) dense matvec (18× fewer operations)
// 512 bytes of sign arrays instead of 256KB of dense matrices
// ===== INLINED turbo-wht.h =====
// TurboQuant Fast Walsh-Hadamard rotation for Metal
// Replaces 256KB dense matrices with 512 bytes of sign arrays + O(d log d) butterfly
// Generated with seed=42 (rotation) and seed=1042 (QJL)

// --- Rotation sign arrays ---
constant float turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
constant float turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// --- Pre-packed half4 sign arrays for vectorized WHT (eliminates float→half conversion) ---
constant half4 turbo_wht_signs1_h4[32] = {
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h)
};
constant half4 turbo_wht_signs2_h4[32] = {
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h)
};

// --- QJL sign arrays ---
constant float turbo_qjl_wht_signs1[128] = {
    1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
constant float turbo_qjl_wht_signs2[128] = {
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f};

// --- Fast Walsh-Hadamard Transform (in-place, normalized) ---
// O(n log n) = 896 operations for n=128, vs O(n²) = 16384 for dense matvec
static void turbo_fwht_128(thread float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    // Normalize by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// --- Forward rotation: signs1 → FWHT → signs2 ---
static void turbo_rotate_forward(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

// --- Inverse rotation: signs2 → FWHT → signs1 (FWHT is its own inverse) ---
static void turbo_rotate_inverse(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
}

// ===== END turbo-wht.h =====

// 2-bit centroids for d=128 (scaled by 1/sqrt(128))
constant float turbo_centroids_2bit[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };
// 3-bit centroids for d=128
constant float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
// Midpoints for 2-bit nearest centroid lookup
constant float turbo_mid_2bit[3] = { -0.086728f, 0.0f, 0.086728f };
// Midpoints for 3-bit
constant float turbo_mid_3bit[7] = { -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f };

// 4-bit PolarQuant centroids (16 levels) — optimal for N(0, 1/sqrt(128))
constant float turbo_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
constant float turbo_mid_4bit[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f,  0.000000f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};

// Half-precision 4-bit centroid LUT for vec path
constant half turbo_centroids_4bit_h[16] = {
    -0.173926h, -0.117195h, -0.089527h, -0.068756h,
    -0.051262h, -0.035597h, -0.020989h, -0.006938h,
     0.006938h,  0.020989h,  0.035597h,  0.051262h,
     0.068756h,  0.089527h,  0.117195h,  0.173926h
};

// 8-entry magnitude LUT for 4-bit (positive half, ascending)
// idx 8-15 are positive: mag = centroids_4bit[idx]
// idx 0-7 are negative: mag = centroids_4bit[15 - idx] with sign flip
// sign = (idx >> 3) ? +1 : -1
// mag_idx = (idx & 7) for positive, (7 - (idx & 7)) for negative — but
// since centroids are symmetric ascending, just use: mag[idx & 7] for idx>=8,
// mag[7 - (idx & 7)] for idx<8. Simpler: mag[idx >= 8 ? idx & 7 : 7 - (idx & 7)]
constant half turbo_mag_4bit_h[8] = {
    0.006938h, 0.020989h, 0.035597h, 0.051262h,
    0.068756h, 0.089527h, 0.117195h, 0.173926h
};

// Half-precision 2-bit centroid LUT for vec path
constant half turbo_centroids_2bit_h[4] = {
    -0.133462h, -0.039994h, 0.039994h, 0.133462h
};

// Quantize 32 elements into one block_turbo2_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
