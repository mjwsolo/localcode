#include "common.h"
#include "dequantize.h"
#include "fa_common.metal"

#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

// TurboQuant KV-cache flash attention. Symmetric K/V follow upstream naming
// (kernel_flash_attn_ext_<type>_...); mixed K/V pairs encode both types as k<K>_v<V>.

// K=turbo2 V=turbo2
template [[host_name("kernel_flash_attn_ext_turbo2_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_turbo2_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 576, 512>;

// K=turbo2 V=turbo3
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo3_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 576, 512>;

// K=turbo2 V=turbo4
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo4_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0, 576, 512>;

// K=turbo2 V=q8_0
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vq8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

// K=turbo3 V=turbo2
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo2_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 576, 512>;

// K=turbo3 V=turbo3
template [[host_name("kernel_flash_attn_ext_turbo3_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_turbo3_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 576, 512>;

// K=turbo3 V=turbo4
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo4_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0, 576, 512>;

// K=turbo3 V=q8_0
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vq8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

// K=turbo4 V=turbo2
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo2_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 576, 512>;

// K=turbo4 V=turbo3
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo3_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 576, 512>;

// K=turbo4 V=turbo4
template [[host_name("kernel_flash_attn_ext_turbo4_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_turbo4_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 576, 512>;

// K=turbo4 V=q8_0
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vq8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

// K=q8_0 V=turbo2
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo2_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 576, 512>;

// K=q8_0 V=turbo3
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo3_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 576, 512>;

// K=q8_0 V=turbo4
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk32_dv32")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 32, 32>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk64_dv64")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 64, 64>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk96_dv96")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 96, 96>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32
