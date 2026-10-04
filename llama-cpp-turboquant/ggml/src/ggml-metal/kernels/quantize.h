#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
    }
    const float d = amax;
    dst.d = d;

    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    for (int j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK2_0; j++) {
        int q = (int)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

void quantize_tq2_0(device const float * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    for (int j = 0; j < QK_K/4; j += 32) {
        for (int m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (int n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = (int)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}

// ===== TurboQuant KV-cache quantizers (fork-local) =====
void quantize_turbo2_0(device const float * src, device block_turbo2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    for (int j = 0; j < QK_TURBO2 / 4; j++) dst.qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_2bit[0]) idx = 0;
        else if (val < turbo_mid_2bit[1]) idx = 1;
        else if (val < turbo_mid_2bit[2]) idx = 2;
        else                              idx = 3;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// Quantize 32 elements into one block_turbo3_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
void quantize_turbo3_0(device const float * src, device block_turbo3_0 & dst) {
#pragma METAL fp math_mode(safe)
    // Compute norm for this 32-element sub-block
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    // Quantize to 3-bit centroids
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst.qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst.signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_3bit[0]) idx = 0;
        else if (val < turbo_mid_3bit[1]) idx = 1;
        else if (val < turbo_mid_3bit[2]) idx = 2;
        else if (val < turbo_mid_3bit[3]) idx = 3;
        else if (val < turbo_mid_3bit[4]) idx = 4;
        else if (val < turbo_mid_3bit[5]) idx = 5;
        else if (val < turbo_mid_3bit[6]) idx = 6;
        else                              idx = 7;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst.signs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_turbo4_0(device const float * src, device block_turbo4_0 & dst) {
#pragma METAL fp math_mode(safe)
    // 4-bit PolarQuant: normalize → rotate → quantize to 16 centroids → nibble pack
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float grp_norm = sqrt(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

    for (int j = 0; j < QK_TURBO4 / 2; j++) dst.qs[j] = 0;

    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float val = x[j];
        uint8_t idx;
        if      (val < turbo_mid_4bit[ 0]) idx = 0;
        else if (val < turbo_mid_4bit[ 1]) idx = 1;
        else if (val < turbo_mid_4bit[ 2]) idx = 2;
        else if (val < turbo_mid_4bit[ 3]) idx = 3;
        else if (val < turbo_mid_4bit[ 4]) idx = 4;
        else if (val < turbo_mid_4bit[ 5]) idx = 5;
        else if (val < turbo_mid_4bit[ 6]) idx = 6;
        else if (val < turbo_mid_4bit[ 7]) idx = 7;
        else if (val < turbo_mid_4bit[ 8]) idx = 8;
        else if (val < turbo_mid_4bit[ 9]) idx = 9;
        else if (val < turbo_mid_4bit[10]) idx = 10;
        else if (val < turbo_mid_4bit[11]) idx = 11;
        else if (val < turbo_mid_4bit[12]) idx = 12;
        else if (val < turbo_mid_4bit[13]) idx = 13;
        else if (val < turbo_mid_4bit[14]) idx = 14;
        else                               idx = 15;

        dst.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        recon_norm_sq += turbo_centroids_4bit[idx] * turbo_centroids_4bit[idx];
    }

    dst.rnorm = half(0.0f);
    float recon_norm = sqrt(recon_norm_sq);
    dst.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
}

