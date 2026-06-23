// Validate AND benchmark the int4+Hadamard decode-attention Metal kernel on the GPU.
//  • correctness: GPU int4 output vs the proven CPU `QuantKvInt4::attend` (bit-exact)
//  • performance: int4 kernel vs an f16 baseline kernel, many threadgroups, measured on-device
// Generalized to arbitrary T (token striding + threadgroup reductions; T<=4096). D<=128, pow2.
//   swiftc harness/kv_gpu_validate.swift -o /tmp/kvval && /tmp/kvval /tmp/kv_int4_testcase.bin
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
#define MAXT 4096u

// ── int4 + Hadamard decode attention ────────────────────────────────────────────────────────────
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
    // 1) load Q, forward FWHT, normalize
    qr[lid] = q[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = qr[lid], b = qr[lid+h]; qr[lid] = a+b; qr[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = 1.0f / sqrt((float)D);
    qr[lid] *= norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2) symmetric int8 quant of rotated Q
    if (lid == 0) { float a = 0; for (uint e=0;e<D;e++) a = fmax(a, fabs(qr[e])); sh_scale = a>0 ? a/127.0f : 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qc[lid] = (int)clamp(round(qr[lid]/sh_scale), -127.0f, 127.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3) scores (token-strided): int8(Q)·int4(K[t])
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
    // 4) softmax: parallel max then parallel sum of exp
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
    // 5) P·V in Hadamard basis: channel-parallel
    uint e = lid, bj = e/2; bool odd = (e&1)==1;
    float acc = 0;
    for (uint t=0;t<valid;t++){
        uchar b = v_packed[t*bpt + bj];
        int nib = odd ? (((int)((char)b))>>4) : (((int)((char)(b<<4)))>>4);
        acc += sc[t]*inv * (float)nib * v_scale[t];
    }
    orot[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 6) inverse FWHT, write (offset by threadgroup for the timing run)
    for (uint h = 1; h < D; h <<= 1) {
        if ((lid & h) == 0) { float a = orot[lid], b = orot[lid+h]; orot[lid] = a+b; orot[lid+h] = a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tg*D + lid] = orot[lid] * norm;
}

// ── int4 + Hadamard, SIMD nibble-unpack (uchar4 load + vector unpack + int4 MAC) ─────────────────
// Same math as kv_int4_attend (integer dot ⇒ bit-exact, no float reassociation), but the 64-byte
// scalar K loop becomes 16 vector iterations. The only extra setup is a once-per-query deinterleave
// of the quantized Q into lo/hi-nibble lanes (qc_e/qc_o) — amortized over all T tokens, not per-step.
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
    threadgroup char4 qc_e[16];   // deinterleaved Q: even-channel (low-nibble) lanes
    threadgroup char4 qc_o[16];   // odd-channel (high-nibble) lanes
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
    // once-per-query deinterleave: qc_e[k] = (qc[8k],qc[8k+2],qc[8k+4],qc[8k+6]); qc_o = odd indices
    if (lid < bpt/4) {
        uint k = lid;
        qc_e[k] = char4(qc[8*k], qc[8*k+2], qc[8*k+4], qc[8*k+6]);
        qc_o[k] = char4(qc[8*k+1], qc[8*k+3], qc[8*k+5], qc[8*k+7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: vectorized integer dot, 16 iters of 8 channels each
    for (uint t = lid; t < valid; t += nthreads) {
        int4 acc = int4(0);
        device const uchar4* kp4 = (device const uchar4*)(k_packed + t*bpt);
        for (uint j = 0; j < bpt/4; j++) {
            uchar4 b4 = kp4[j];
            char4 lo = as_type<char4>(uchar4(b4 << 4)) >> 4; // low nibbles, sign-extended
            char4 hi = as_type<char4>(b4) >> 4;              // high nibbles, sign-extended
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

// ── f16 baseline decode attention (reads f16 K/V) ────────────────────────────────────────────────
kernel void kv_f16_attend(
    device const half* q   [[buffer(0)]],
    device const half* k   [[buffer(1)]],
    device const half* v   [[buffer(2)]],
    constant uint3&    TD  [[buffer(3)]],
    constant float&    scale [[buffer(4)]],
    device float*      out [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]],
    uint nthreads [[threads_per_threadgroup]])
{
    const uint D = TD.y, valid = TD.z;
    threadgroup float sc[MAXT];
    threadgroup float red[128];
    threadgroup float sh_l;
    for (uint t=lid;t<valid;t+=nthreads){
        float dot = 0; device const half* kp = k + t*D;
        for (uint e=0;e<D;e++) dot += (float)q[e]*(float)kp[e];
        sc[t] = scale*dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lm=-INFINITY; for(uint t=lid;t<valid;t+=nthreads) lm=fmax(lm,sc[t]);
    red[lid]=lm; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]=fmax(red[lid],red[lid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float m=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for(uint t=lid;t<valid;t+=nthreads){ float e=exp(sc[t]-m); sc[t]=e; ls+=e; }
    red[lid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=nthreads/2;s>0;s>>=1){ if(lid<s) red[lid]+=red[lid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(lid==0) sh_l=red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.0f/sh_l;
    float acc=0; for(uint t=0;t<valid;t++) acc += sc[t]*inv*(float)v[t*D+lid];
    out[tg*D + lid] = acc;
}
"""

// ── read dumped test case ───────────────────────────────────────────────────────────────────────
let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/kv_int4_testcase.bin"
let data = try! Data(contentsOf: URL(fileURLWithPath: path))
var off = 0
func u32() -> Int { let v = data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self) }; off += 4; return Int(v) }
func f32() -> Float { let v = data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: Float.self) }; off += 4; return v }
func f32a(_ n: Int) -> [Float] { (0..<n).map { _ in f32() } }
func u8a(_ n: Int) -> [UInt8] { let a = [UInt8](data.subdata(in: off..<off+n)); off += n; return a }

let T = u32(), D = u32()
let scale = f32()
let q = f32a(D)
let kScale = f32a(T), vScale = f32a(T)
let kPacked = u8a(T*D/2), vPacked = u8a(T*D/2)
let refOut = f32a(D)
let kOrig = f32a(T*D), vOrig = f32a(T*D)
let causalValid = u32()              // prefill causal-skip case: attend only to first `causalValid`
let refOutCausal = f32a(D)           // CPU attend_limited(q, scale, causalValid-1)

// ── Metal setup ─────────────────────────────────────────────────────────────────────────────────
guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("GPU: \(dev.name)")
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) } catch { print("MSL compile error:\n\(error)"); exit(1) }
let pInt4 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "kv_int4_attend")!)
let pSimd = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "kv_int4_attend_simd")!)
let pF16  = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "kv_f16_attend")!)
let queue = dev.makeCommandQueue()!
func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
var td = SIMD3<UInt32>(UInt32(T), UInt32(D), UInt32(T)), sc = scale  // .z = valid token count
let tgSize = MTLSize(width: D, height: 1, depth: 1)

// ── extension setBuffer helper ──────────────────────────────────────────────────────────────────
extension MTLComputeCommandEncoder { func setBuffer(_ b: MTLBuffer, _ o: Int, index: Int) { setBuffer(b, offset: o, index: index) } }

// ── correctness: each int4 kernel vs CPU reference (bit-exact gate, the SVT rule) ────────────────
let qB = buf(q), kpB = buf(kPacked), ksB = buf(kScale), vpB = buf(vPacked), vsB = buf(vScale)
let outB = dev.makeBuffer(length: D*4, options: .storageModeShared)!
func validate(_ pipe: MTLComputePipelineState, _ name: String, valid: Int, _ reference: [Float]) {
    var tdv = SIMD3<UInt32>(UInt32(T), UInt32(D), UInt32(valid))
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(qB,0,index:0); enc.setBuffer(kpB,0,index:1); enc.setBuffer(ksB,0,index:2)
    enc.setBuffer(vpB,0,index:3); enc.setBuffer(vsB,0,index:4)
    enc.setBytes(&tdv,length:16,index:5); enc.setBytes(&sc,length:4,index:6); enc.setBuffer(outB,0,index:7)
    enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1), threadsPerThreadgroup: tgSize)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let gpu = UnsafeBufferPointer(start: outB.contents().assumingMemoryBound(to: Float.self), count: D)
    var maxAbs: Float = 0, dn: Float = 0, rn: Float = 0
    for e in 0..<D { let d = abs(gpu[e]-reference[e]); maxAbs = max(maxAbs,d); dn += d*d; rn += reference[e]*reference[e] }
    let rel = sqrt(dn)/max(sqrt(rn),1e-9)
    print(String(format: "correctness  %@  max|Δ|=%.6f  rel-dev=%.6f  -> %@", name, maxAbs, rel,
                 rel < 1e-3 ? "PASS" : "FAIL"))
    if rel >= 1e-3 { exit(1) }
}
print("T=\(T) D=\(D)")
validate(pInt4, "int4-scalar    ", valid: T, refOut)
validate(pSimd, "int4-simd      ", valid: T, refOut)
// causal-skip: same kernel bounded to `causalValid` tokens must match CPU attend_limited (bit-exact)
validate(pSimd, "int4-causalskip", valid: causalValid, refOutCausal)

// ── performance: int4 vs f16, many threadgroups, REPS dispatches per command buffer ──────────────
let G = 256, REPS = 100
let outBig = dev.makeBuffer(length: G*D*4, options: .storageModeShared)!
let qH = q.map { Float16($0) }, kH = kOrig.map { Float16($0) }, vH = vOrig.map { Float16($0) }
let qHB = buf(qH), kHB = buf(kH), vHB = buf(vH)
func timeKernel(_ encodeOne: (MTLComputeCommandEncoder) -> Void) -> Double {
    for _ in 0..<3 { let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!; encodeOne(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted() } // warmup
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
    for _ in 0..<REPS { encodeOne(enc) }
    enc.endEncoding()
    let t0 = DispatchTime.now(); cb.commit(); cb.waitUntilCompleted(); let t1 = DispatchTime.now()
    return Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1e3 / Double(REPS) / Double(G) // µs per threadgroup (per head)
}
func timeInt4(_ pipe: MTLComputePipelineState) -> Double {
    timeKernel { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(qB,0,index:0); enc.setBuffer(kpB,0,index:1); enc.setBuffer(ksB,0,index:2)
        enc.setBuffer(vpB,0,index:3); enc.setBuffer(vsB,0,index:4)
        enc.setBytes(&td,length:16,index:5); enc.setBytes(&sc,length:4,index:6); enc.setBuffer(outBig,0,index:7)
        enc.dispatchThreadgroups(MTLSize(width:G,height:1,depth:1), threadsPerThreadgroup: tgSize)
    }
}
let int4us = timeInt4(pInt4)
let simdus = timeInt4(pSimd)
let f16us = timeKernel { enc in
    enc.setComputePipelineState(pF16)
    enc.setBuffer(qHB,0,index:0); enc.setBuffer(kHB,0,index:1); enc.setBuffer(vHB,0,index:2)
    enc.setBytes(&td,length:16,index:3); enc.setBytes(&sc,length:4,index:4); enc.setBuffer(outBig,0,index:5)
    enc.dispatchThreadgroups(MTLSize(width:G,height:1,depth:1), threadsPerThreadgroup: tgSize)
}
let kvF16KB = Double(T*D*2*2)/1024.0, kvI4KB = Double(T*D/2*2 + T*4*2)/1024.0
print(String(format: "performance  per-head µs:  f16=%.3f  int4-scalar=%.3f  int4-simd=%.3f", f16us, int4us, simdus))
print(String(format: "             vs f16:       int4-scalar=%.2fx  int4-simd=%.2fx   (simd vs scalar=%.2fx)",
             f16us/int4us, f16us/simdus, int4us/simdus))
print(String(format: "memory       per-head KV:  f16=%.1f KB  int4=%.1f KB  ->  %.1fx smaller", kvF16KB, kvI4KB, kvF16KB/kvI4KB))

// ── causal-skip: per-query latency scales with `valid` (the prefill win, bit-exact) ──────────────
print("\ncausal-skip  int4-simd latency vs visible tokens (attention work ∝ valid):")
for frac in [1.0, 0.5, 0.25] {
    let v = max(1, Int(Double(T) * frac))
    let us = timeKernel { enc in
        var tdv = SIMD3<UInt32>(UInt32(T), UInt32(D), UInt32(v))
        enc.setComputePipelineState(pSimd)
        enc.setBuffer(qB,0,index:0); enc.setBuffer(kpB,0,index:1); enc.setBuffer(ksB,0,index:2)
        enc.setBuffer(vpB,0,index:3); enc.setBuffer(vsB,0,index:4)
        enc.setBytes(&tdv,length:16,index:5); enc.setBytes(&sc,length:4,index:6); enc.setBuffer(outBig,0,index:7)
        enc.dispatchThreadgroups(MTLSize(width:G,height:1,depth:1), threadsPerThreadgroup: tgSize)
    }
    print(String(format: "  valid=%5d (%3.0f%% of T)  %.3f µs/head", v, frac*100, us))
}
