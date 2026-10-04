#include "common.h"
#include "dequantize.h"
#include "fa_vec_common.metal"

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// TurboQuant KV-cache vec flash attention: baseline (Q=1, NE=fa_vec_baseline_ne) only.
// ggml_metal_tuning::fa_vec_pick has no tuned entries for these dtypes, so it always returns the baseline.

// K=turbo2 V=turbo2
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// K=turbo2 V=turbo3
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// K=turbo2 V=turbo4
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// K=turbo2 V=q8_0
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

// K=turbo3 V=turbo2
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// K=turbo3 V=turbo3
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// K=turbo3 V=turbo4
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// K=turbo3 V=q8_0
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

// K=turbo4 V=turbo2
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// K=turbo4 V=turbo3
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// K=turbo4 V=turbo4
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_turbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// K=turbo4 V=q8_0
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

// K=q8_0 V=turbo2
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// K=q8_0 V=turbo3
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// K=q8_0 V=turbo4
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

#undef FA_TYPES
#undef FA_TYPES_F32
