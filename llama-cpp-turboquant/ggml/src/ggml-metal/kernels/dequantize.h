#pragma once

#include "common.h"

#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
#include "ggml-common.h"
#endif

#define QK_NL 16 // shared by mul_mm and get_rows_q instantiations

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#if defined(GGML_METAL_HAS_BF16)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_bf16_t4(device const bfloat4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#endif

template <typename type4x4>
void dequantize_q1_0(device const block_q1_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;
    const float neg_d = -d;

    const int byte_offset = il * 2;  // il*16 bits = il*2 bytes
    const uint8_t b0 = qs[byte_offset];
    const uint8_t b1 = qs[byte_offset + 1];

    float4x4 reg_f;

    reg_f[0][0] = select(neg_d, d, bool(b0 & 0x01));
    reg_f[0][1] = select(neg_d, d, bool(b0 & 0x02));
    reg_f[0][2] = select(neg_d, d, bool(b0 & 0x04));
    reg_f[0][3] = select(neg_d, d, bool(b0 & 0x08));
    reg_f[1][0] = select(neg_d, d, bool(b0 & 0x10));
    reg_f[1][1] = select(neg_d, d, bool(b0 & 0x20));
    reg_f[1][2] = select(neg_d, d, bool(b0 & 0x40));
    reg_f[1][3] = select(neg_d, d, bool(b0 & 0x80));

    reg_f[2][0] = select(neg_d, d, bool(b1 & 0x01));
    reg_f[2][1] = select(neg_d, d, bool(b1 & 0x02));
    reg_f[2][2] = select(neg_d, d, bool(b1 & 0x04));
    reg_f[2][3] = select(neg_d, d, bool(b1 & 0x08));
    reg_f[3][0] = select(neg_d, d, bool(b1 & 0x10));
    reg_f[3][1] = select(neg_d, d, bool(b1 & 0x20));
    reg_f[3][2] = select(neg_d, d, bool(b1 & 0x40));
    reg_f[3][3] = select(neg_d, d, bool(b1 & 0x80));

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q1_0_t4(device const block_q1_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const float neg_d = -d;
    const int base = il * 4;
    const uint8_t byte = xb->qs[base / 8];
    const int s = base % 8;

    float4 reg_f;
    reg_f[0] = select(neg_d, d, bool((byte >> (s    )) & 1));
    reg_f[1] = select(neg_d, d, bool((byte >> (s + 1)) & 1));
    reg_f[2] = select(neg_d, d, bool((byte >> (s + 2)) & 1));
    reg_f[3] = select(neg_d, d, bool((byte >> (s + 3)) & 1));

    reg = (type4) reg_f;
}

template <typename type4x4>
void dequantize_q2_0(device const block_q2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    const int byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const uint8_t b = qs[byte_offset + i];
        reg_f[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q2_0_t4(device const block_q2_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const uint8_t b = xb->qs[il];

    float4 reg_f;
    reg_f[0] = ((float)((b >> 0) & 3) - 1.0f) * d;
    reg_f[1] = ((float)((b >> 2) & 3) - 1.0f) * d;
    reg_f[2] = ((float)((b >> 4) & 3) - 1.0f) * d;
    reg_f[3] = ((float)((b >> 6) & 3) - 1.0f) * d;

    reg = (type4) reg_f;
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}



template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_mxfp4_t4(device const block_mxfp4 * xb, short il, thread type4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const short il4 = il%4;

    const uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_tq2_0(device const block_tq2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    // 2 bits per element, 4 elements per byte, 128 elements per 32-byte group
    const short base = il * 16;
    for (int k = 0; k < 16; k++) {
        const int i = base + k;
        const int byte = ((i >> 7) & 1) * 32 + (i & 31);
        const int l = (i >> 5) & 3;
        reg_f[k/4][k%4] = d * (float)(((qs[byte] >> (2*l)) & 3) - 1);
    }

    reg = (type4x4) reg_f;
}

// ===== TurboQuant KV-cache dequantizers (fork-local) =====
// ----- turbo3 dequantize with per-thread block cache -----
// The rotation requires all 128 elements. Flash attention calls dequantize
// up to 32× per block (once per 4-element chunk). We cache the full
// dequantized block per thread and only recompute when the block pointer changes.

// turbo3 dequant — full block dequantize with inverse rotation
// Must process all 128 elements to apply WHT inverse rotation
// Half-precision vectorized WHT for faster dequant.
// Uses half4 vectors for 4-wide SIMD throughput on Apple GPU.
// Centroids fit in fp16 (max |val| = 0.19), butterfly stays in range.
static void turbo_fwht_128_half4(thread half4 * v) {
    // 32 half4 vectors = 128 elements
    // Stage h=1: butterfly between elements 0,1 and 2,3 within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // Stage h=2: butterfly between elements 0,2 and 1,3 within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // Stages h=4,8,16,32,64: butterfly between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;  // distance in half4 units
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i];
                half4 b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }
    // Normalize
    const half4 inv_sqrt_128 = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        v[i] *= inv_sqrt_128;
    }
}

// ----- turbo2 dequantize -----
// 2-bit indices (4 centroids), no signs byte. Simpler than turbo3.
// Block size 32, nl=2 for non-vec (32/16), nl=8 for vec (32/4).

// Non-vec: 16 elements per call (il ∈ {0,1}), returns type4x4
template <typename type4x4>
void dequantize_turbo2_0(device const block_turbo2_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    // il=0 → elements 0-15 (qs bytes 0-3)
    // il=1 → elements 16-31 (qs bytes 4-7)
    const int qs_off = il * 4;
    float4x4 reg_f;
    for (int g = 0; g < 4; g++) {
        const uint8_t qb = xb->qs[qs_off + g];
        reg_f[g] = float4(
            turbo_centroids_2bit[(qb      ) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 2) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 4) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 6)       ] * norm
        );
    }
    reg = (type4x4) reg_f;
}

// Vec: 4 elements per call (il ∈ {0..7}), returns type4
template <typename type4>
void dequantize_turbo2_0_t4(device const block_turbo2_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    // il selects which byte of qs (each byte has 4 x 2-bit values)
    const uint8_t qb = xb->qs[il];
    reg = type4(float4(
        float(turbo_centroids_2bit_h[(qb      ) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 2) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 4) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 6)       ]) * norm
    ));
}

// Block-32 dequant: no WHT needed (graph handles rotation). Just centroid lookup + norm scale.
// With QK_TURBO3=32: nl=2 for non-vec FA (32/16), nl=8 for vec FA (32/4).
// Much less redundant work than block-128.

// Optimized turbo3 dequant: batch byte reads, unrolled index extraction.
// Non-vec: 16 elements per call (il ∈ {0,1}), returns type4x4
template <typename type4x4>
void dequantize_turbo3_0(device const block_turbo3_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    // il=0 → elements 0-15 (qs bytes 0-3, signs bytes 0-1)
    // il=1 → elements 16-31 (qs bytes 4-7, signs bytes 2-3)
    const int qs_off = il * 4;
    float4x4 reg_f;
    for (int g = 0; g < 4; g++) {
        // g iterates over 4 groups of 4 elements within our 16
        // element index within block: il*16 + g*4 + k, k=0..3
        const uint8_t qb = xb->qs[qs_off + g];
        // signs byte index: (il*16 + g*4) / 8 = il*2 + g/2
        const uint8_t sb = xb->signs[il * 2 + g / 2];
        const int sshift = (g & 1) * 4;

        reg_f[g] = float4(
            turbo_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)] * norm
        );
    }
    reg = (type4x4) reg_f;
}

// Half-precision centroid LUT for vec path — reduces constant cache pressure at long context.
// Register LUT (cn[8] = centroid*norm in thread registers) was tested but caused register
// spill on Metal, making it slower. Constant half LUT + float norm broadcast remains the
// fastest approach on Apple Silicon. On CUDA, register LUT works better (see @spiritbuun).
constant half turbo_centroids_3bit_h[8] = {
    -0.190685h, -0.117832h, -0.065717h, -0.021460h,
     0.021460h,  0.065717h,  0.117832h,  0.190685h
};

// 4-entry magnitude LUT (positive values only, ascending order)
// Used with ALU sign application to halve constant cache divergence
constant half turbo_mag_3bit_h[4] = {
    0.021460h, 0.065717h, 0.117832h, 0.190685h
};

// 2-entry PAIR LUT: each entry is a half2 containing two adjacent magnitudes.
// Only 2 possible constant addresses per lookup (vs 4 for mag LUT, 8 for full).
// bit1 selects the pair, bit0 selects within the pair via ternary.
constant half2 turbo_mag_pairs_h[2] = {
    half2(0.021460h, 0.065717h),   // pair 0: mag indices 0,1
    half2(0.117832h, 0.190685h),   // pair 1: mag indices 2,3
};

// Vec: 4 elements per call (il ∈ {0..7}), returns type4
// Experiment: batched byte reads (ported from @spiritbuun's CUDA impl).
// Read qs + signs bytes with minimal device memory accesses.
// The signs byte covers 8 elements — read once, shift for each element.
// Constant half[8] LUT for centroid lookup (proven fastest on Apple Silicon).
template <typename type4>
void dequantize_turbo3_0_t4(device const block_turbo3_0 * xb, short il, thread type4 & reg) {
    // PROFILING MODE: controlled by TURBO_PROFILE_MODE compile flag
    // 0 = full dequant (batched extract)
    // 1 = no-op (return zeros) — decode ceiling without dequant cost
    // 2 = norm only (read norm, return constant) — isolate norm read
    // 3 = norm + qs only (skip signs) — isolate signs byte cost
    // 4 = full dequant, skip LUT (use constant centroid) — isolate LUT cost
#ifndef TURBO_PROFILE_MODE
#define TURBO_PROFILE_MODE 0
#endif

#if TURBO_PROFILE_MODE == 1
    // NO-OP: decode speed ceiling
    reg = type4(0.0f);
#elif TURBO_PROFILE_MODE == 2
    // NORM ONLY: just read norm, return it as all 4 values
    const float norm = float(xb->norm);
    reg = type4(norm);
#elif TURBO_PROFILE_MODE == 3
    // NORM + QS: read norm and qs byte, skip signs
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t q0 = (qb      ) & 0x03;
    const uint8_t q1 = (qb >> 2) & 0x03;
    const uint8_t q2 = (qb >> 4) & 0x03;
    const uint8_t q3 = (qb >> 6);
    // Use qs without signs — just positive centroids
    reg = type4(float4(
        float(turbo_centroids_3bit_h[q0 + 4]),
        float(turbo_centroids_3bit_h[q1 + 4]),
        float(turbo_centroids_3bit_h[q2 + 4]),
        float(turbo_centroids_3bit_h[q3 + 4])
    ) * norm);
#elif TURBO_PROFILE_MODE == 4
    // SKIP LUT: read all bytes but use constant centroid value
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il >> 1];
    // Pretend all elements are centroid 0 — isolates LUT indexing cost
    reg = type4(float4(float(turbo_centroids_3bit_h[0])) * norm);
#else
    // MODE 0: 4-entry magnitude LUT + ALU sign (halves constant cache divergence)
    // Only 4 possible constant addresses per lookup (vs 8 in full LUT).
    // Sign applied via select() — no branch, just conditional negate.
    // Correct sign mapping: sign=1 → +mag[qs], sign=0 → -mag[3-qs] (reversed)
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il >> 1];
    const int sshift = (il & 1) << 2;

    const uint8_t q0 = (qb      ) & 0x03;
    const uint8_t q1 = (qb >> 2) & 0x03;
    const uint8_t q2 = (qb >> 4) & 0x03;
    const uint8_t q3 = (qb >> 6);
    const uint8_t s0 = (sb >> (sshift    )) & 1;
    const uint8_t s1 = (sb >> (sshift + 1)) & 1;
    const uint8_t s2 = (sb >> (sshift + 2)) & 1;
    const uint8_t s3 = (sb >> (sshift + 3)) & 1;

    // Auto-selected dequant path based on hardware.
    // TURBO_USE_4MAG=1 (pre-M5): 4-entry magnitude LUT + XOR sign (+38-45% on M2)
    // TURBO_USE_4MAG=0 (M5+): 8-entry full LUT (best on M5, 0.905x q8_0)
#if TURBO_USE_4MAG
    // 4-mag LUT (proven +38-45% on M2). See decode-speed-hardware-analysis.md
    // for the full 12-approach experiment log.
    const uint8_t mi0 = q0 ^ (s0 ? 0u : 0x3u);
    const uint8_t mi1 = q1 ^ (s1 ? 0u : 0x3u);
    const uint8_t mi2 = q2 ^ (s2 ? 0u : 0x3u);
    const uint8_t mi3 = q3 ^ (s3 ? 0u : 0x3u);

    const float v0 = float(turbo_mag_3bit_h[mi0]) * norm;
    const float v1 = float(turbo_mag_3bit_h[mi1]) * norm;
    const float v2 = float(turbo_mag_3bit_h[mi2]) * norm;
    const float v3 = float(turbo_mag_3bit_h[mi3]) * norm;

    reg = type4(float4(
        s0 ? v0 : -v0,
        s1 ? v1 : -v1,
        s2 ? v2 : -v2,
        s3 ? v3 : -v3
    ));
#else
    // 8-entry full LUT: best on M5 Max (0.905x q8_0, 77.4 tok/s)
    reg = type4(float4(
        float(turbo_centroids_3bit_h[q0 | (s0 << 2)]),
        float(turbo_centroids_3bit_h[q1 | (s1 << 2)]),
        float(turbo_centroids_3bit_h[q2 | (s2 << 2)]),
        float(turbo_centroids_3bit_h[q3 | (s3 << 2)])
    ) * norm);
#endif
#endif
}

// ----- turbo4 dequantize with per-thread block cache -----

static void turbo4_dequantize_full_block(device const block_turbo4_0 * xb, thread float * cache) {
    const float norm = float(xb->norm);

    // 4-bit nibble unpack — 2 elements per byte, simple and fast
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
        cache[j] = turbo_centroids_4bit[idx] * norm;
    }
}

template <typename type4x4>
void dequantize_turbo4_0(device const block_turbo4_0 * xb, short il, thread type4x4 & reg) {
    // Direct 16-element extraction — 4-bit nibble unpack
    const float norm = float(xb->norm);
    const int base = il * 16;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        for (int k = 0; k < 4; k++) {
            const int j = base + g * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            reg_f[g][k] = turbo_centroids_4bit[idx] * norm;
        }
    }
    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_turbo4_0_t4(device const block_turbo4_0 * xb, short il, thread type4 & reg) {
    // Direct 16-entry half LUT — fastest on M5 Max (constant cache not the bottleneck)
    // 8-mag LUT tested: -3% on M5 due to ternary branch overhead. Keep for M1/M2 if needed.
    const float norm = float(xb->norm);
    const device uint8_t * qs = xb->qs + il * 2;
    const uint8_t qb0 = qs[0];
    const uint8_t qb1 = qs[1];

    reg = type4(float4(
        float(turbo_centroids_4bit_h[(qb0     ) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb0 >> 4) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb1     ) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb1 >> 4) & 0xF]) * norm
    ));
}

// ============================================================================
// TQ3_1S / TQ4_1S: WHT-rotated weight quantization constants and dequant
// ============================================================================

constant float tq3_centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f,
};

constant float tq4_centroids[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

constant float tq3_signs[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

constant float tq3_inv_sqrt32 = 0.17677669529663688f; // 1/sqrt(32)

// Unpack 32 x 3-bit indices from 12 bytes (4 groups of 3 bytes = 8 indices each)
inline void tq3_unpack_indices(device const uint8_t * qs, thread uint8_t * indices) {
    for (int g = 0; g < 4; g++) {
        device const uint8_t * qp = qs + g * 3;
        thread uint8_t * idx = indices + g * 8;
        idx[0] =  qp[0]       & 7;
        idx[1] = (qp[0] >> 3) & 7;
        idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        idx[3] = (qp[1] >> 1) & 7;
        idx[4] = (qp[1] >> 4) & 7;
        idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        idx[6] = (qp[2] >> 2) & 7;
        idx[7] = (qp[2] >> 5) & 7;
    }
}

// Unpack 8 indices from a single 3-byte group
inline void tq3_unpack_group(device const uint8_t * qp, thread uint8_t * idx) {
    idx[0] =  qp[0]       & 7;
    idx[1] = (qp[0] >> 3) & 7;
    idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
    idx[3] = (qp[1] >> 1) & 7;
    idx[4] = (qp[1] >> 4) & 7;
    idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
    idx[6] = (qp[2] >> 2) & 7;
    idx[7] = (qp[2] >> 5) & 7;
}

// In-place inverse RHT: WHT butterfly + normalize + undo sign flips
inline void tq3_rht_inverse(thread float * buf) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; i++) {
        buf[i] *= tq3_inv_sqrt32 * tq3_signs[i];
    }
}

// TQ3_1S dequant: 4x4 with inverse RHT
template <typename type4x4>
void dequantize_tq3_1s(device const block_tq3_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    uint8_t indices[32];
    tq3_unpack_indices(xb->qs, indices);

    float buf[32];
    for (int j = 0; j < 16; j++) buf[j]  = tq3_centroids[indices[j]]  * d0;
    for (int j = 16; j < 32; j++) buf[j] = tq3_centroids[indices[j]] * d1;

    tq3_rht_inverse(buf);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            reg_f[i][k] = buf[il * 16 + i * 4 + k];
        }
    }
    reg = (type4x4) reg_f;
}

// TQ3_1S dequant: t4 with inverse RHT
template <typename type4>
void dequantize_tq3_1s_t4(device const block_tq3_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    uint8_t indices[32];
    tq3_unpack_indices(xb->qs, indices);

    float buf[32];
    for (int j = 0; j < 16; j++) buf[j]  = tq3_centroids[indices[j]]  * d0;
    for (int j = 16; j < 32; j++) buf[j] = tq3_centroids[indices[j]] * d1;

    tq3_rht_inverse(buf);

    int off = il * 4;
    reg = (type4) float4(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
}

// TQ4_1S dequant: 4x4 with inverse RHT
template <typename type4x4>
void dequantize_tq4_1s(device const block_tq4_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = tq4_centroids[idx] * d;
    }

    tq3_rht_inverse(buf);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            reg_f[i][k] = buf[il * 16 + i * 4 + k];
        }
    }
    reg = (type4x4) reg_f;
}

// TQ4_1S dequant: t4 with inverse RHT
template <typename type4>
void dequantize_tq4_1s_t4(device const block_tq4_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = tq4_centroids[idx] * d;
    }

    tq3_rht_inverse(buf);

    int off = il * 4;
    reg = (type4) float4(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
}

