// int4 + Hadamard KV-cache decode attention (Milestone-4 kernel), ported into tract-metal.
// Validated bit-exact vs the CPU `QuantKvInt4::attend` on an Apple M4 (harness/kv_gpu_validate.swift).
// One threadgroup per (batch, query head); THREADS == D (power of two, ≤128). TD = (T, D, valid).
// `valid` (TD.z) bounds the attended tokens to 0..valid — the bit-exact causal-skip: a prefill query
// at position i passes valid=i+1, so its attention work is ∝ i (not T), halving the prompt's
// attention compute. Two variants: scalar nibble-unpack and SIMD (uchar4 load + vector unpack).
#include <metal_stdlib>
using namespace metal;
#define MAXT 4096u

kernel void kv_int4_attend(
    device const float* q        [[buffer(0)]],
    device const uchar* k_packed [[buffer(1)]],
    device const float* k_scale  [[buffer(2)]],
    device const uchar* v_packed [[buffer(3)]],
    device const float* v_scale  [[buffer(4)]],
    constant uint3&     TD       [[buffer(5)]],
    constant float&     scale    [[buffer(6)]],
    device float*       out      [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]],
    uint nthreads [[threads_per_threadgroup]])
{
    const uint D = TD.y, valid = TD.z, bpt = D/2;
    threadgroup float qr[128];
    threadgroup int   qc[128];
    threadgroup float sc[MAXT];
    threadgroup float orot[128];
    threadgroup float red[128];
    threadgroup float sh_scale, sh_l;
    qr[lid] = q[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = qr[lid], b = qr[lid+h]; qr[lid] = a+b; qr[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = 1.0f / sqrt((float)D);
    qr[lid] *= norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) { float a = 0; for (uint e=0;e<D;e++) a = fmax(a, fabs(qr[e])); sh_scale = a>0 ? a/127.0f : 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qc[lid] = (int)clamp(round(qr[lid]/sh_scale), -127.0f, 127.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = lid; t < valid; t += nthreads) {
        int dot = 0;
        device const uchar* kp = k_packed + t*bpt;
        for (uint j = 0; j < bpt; j++) {
            uchar b = kp[j];
            int lo = ((int)((char)(b<<4)))>>4, hi = ((int)((char)b))>>4;
            dot += qc[2*j]*lo + qc[2*j+1]*hi;
        }
        sc[t] = scale * sh_scale * k_scale[t] * (float)dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lm = -INFINITY; for (uint t=lid;t<valid;t+=nthreads) lm = fmax(lm, sc[t]);
    red[lid] = lm; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]=fmax(red[lid],red[lid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float m = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls = 0; for (uint t=lid;t<valid;t+=nthreads){ float e=exp(sc[t]-m); sc[t]=e; ls+=e; }
    red[lid] = ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]+=red[lid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (lid==0) sh_l = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sh_l;
    uint e = lid, bj = e/2; bool odd = (e&1)==1;
    float acc = 0;
    for (uint t=0;t<valid;t++){
        uchar b = v_packed[t*bpt + bj];
        int nib = odd ? (((int)((char)b))>>4) : (((int)((char)(b<<4)))>>4);
        acc += sc[t]*inv * (float)nib * v_scale[t];
    }
    orot[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = orot[lid], b = orot[lid+h]; orot[lid] = a+b; orot[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tg*D + lid] = orot[lid] * norm;
}

kernel void kv_int4_attend_simd(
    device const float* q        [[buffer(0)]],
    device const uchar* k_packed [[buffer(1)]],
    device const float* k_scale  [[buffer(2)]],
    device const uchar* v_packed [[buffer(3)]],
    device const float* v_scale  [[buffer(4)]],
    constant uint3&     TD       [[buffer(5)]],
    constant float&     scale    [[buffer(6)]],
    device float*       out      [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]],
    uint nthreads [[threads_per_threadgroup]])
{
    const uint D = TD.y, valid = TD.z, bpt = D/2;
    threadgroup float qr[128];
    threadgroup int   qc[128];
    threadgroup char4 qc_e[16];
    threadgroup char4 qc_o[16];
    threadgroup float sc[MAXT];
    threadgroup float orot[128];
    threadgroup float red[128];
    threadgroup float sh_scale, sh_l;
    qr[lid] = q[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = qr[lid], b = qr[lid+h]; qr[lid] = a+b; qr[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = 1.0f / sqrt((float)D);
    qr[lid] *= norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) { float a = 0; for (uint e=0;e<D;e++) a = fmax(a, fabs(qr[e])); sh_scale = a>0 ? a/127.0f : 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qc[lid] = (int)clamp(round(qr[lid]/sh_scale), -127.0f, 127.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < bpt/4) {
        uint k = lid;
        qc_e[k] = char4(qc[8*k], qc[8*k+2], qc[8*k+4], qc[8*k+6]);
        qc_o[k] = char4(qc[8*k+1], qc[8*k+3], qc[8*k+5], qc[8*k+7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = lid; t < valid; t += nthreads) {
        int4 acc = int4(0);
        device const uchar4* kp4 = (device const uchar4*)(k_packed + t*bpt);
        for (uint j = 0; j < bpt/4; j++) {
            uchar4 b4 = kp4[j];
            char4 lo = as_type<char4>(uchar4(b4 << 4)) >> 4;
            char4 hi = as_type<char4>(b4) >> 4;
            acc += int4(lo) * int4(qc_e[j]) + int4(hi) * int4(qc_o[j]);
        }
        int dot = acc.x + acc.y + acc.z + acc.w;
        sc[t] = scale * sh_scale * k_scale[t] * (float)dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lm = -INFINITY; for (uint t=lid;t<valid;t+=nthreads) lm = fmax(lm, sc[t]);
    red[lid] = lm; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]=fmax(red[lid],red[lid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float m = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls = 0; for (uint t=lid;t<valid;t+=nthreads){ float e=exp(sc[t]-m); sc[t]=e; ls+=e; }
    red[lid] = ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]+=red[lid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (lid==0) sh_l = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sh_l;
    uint e = lid, bj = e/2; bool odd = (e&1)==1;
    float acc = 0;
    for (uint t=0;t<valid;t++){
        uchar b = v_packed[t*bpt + bj];
        int nib = odd ? (((int)((char)b))>>4) : (((int)((char)(b<<4)))>>4);
        acc += sc[t]*inv * (float)nib * v_scale[t];
    }
    orot[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = orot[lid], b = orot[lid+h]; orot[lid] = a+b; orot[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tg*D + lid] = orot[lid] * norm;
}
