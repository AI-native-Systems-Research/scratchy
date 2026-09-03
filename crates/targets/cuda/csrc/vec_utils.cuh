// SPDX-License-Identifier: Apache-2.0
// Vectorized load/store helpers for CUDA kernels.
//
// Provides type traits mapping scalar types to their widest aligned vector
// types (128-bit loads), plus cast helpers for processing in float.
//
// For float:       float4 (4 elements, 16 bytes)
// For half/bf16:   uint4  (8 elements, 16 bytes each via reinterpret)

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Type traits: scalar T -> vectorized type for 128-bit loads
// ---------------------------------------------------------------------------

template <typename T>
struct VecType {};

template <>
struct VecType<float> {
    using Type = float4;
    static constexpr int SIZE = 4;  // elements per vector
};

template <>
struct VecType<__half> {
    using Type = uint4;
    static constexpr int SIZE = 8;  // 16 bytes / 2 bytes per half
};

template <>
struct VecType<__nv_bfloat16> {
    using Type = uint4;
    static constexpr int SIZE = 8;
};

// ---------------------------------------------------------------------------
// Vector load/store helpers
// ---------------------------------------------------------------------------

// Load VEC_SIZE elements starting at ptr as a single vector load.
template <typename T>
__device__ __forceinline__ typename VecType<T>::Type vec_load(const T* ptr) {
    return *reinterpret_cast<const typename VecType<T>::Type*>(ptr);
}

// Store VEC_SIZE elements to ptr as a single vector store.
template <typename T>
__device__ __forceinline__ void vec_store(T* ptr, typename VecType<T>::Type val) {
    *reinterpret_cast<typename VecType<T>::Type*>(ptr) = val;
}

// ---------------------------------------------------------------------------
// Element access for float4 vectors
// ---------------------------------------------------------------------------

__device__ __forceinline__ float vec_elem(float4 v, int i) {
    return ((float*)&v)[i];
}

__device__ __forceinline__ void vec_set(float4& v, int i, float val) {
    ((float*)&v)[i] = val;
}

// ---------------------------------------------------------------------------
// Conversion helpers: vector -> float array, float array -> vector
// ---------------------------------------------------------------------------

// Unpack a float4 into a float array.
__device__ __forceinline__ void vec_to_float(float4 v, float* out) {
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

__device__ __forceinline__ float4 float_to_vec_f32(const float* in) {
    return make_float4(in[0], in[1], in[2], in[3]);
}

// Unpack a uint4 containing 8 halves into a float array.
__device__ __forceinline__ void vec_to_float(uint4 v, float* out, const __half* /*tag*/) {
    __half* h = reinterpret_cast<__half*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __half2float(h[i]);
    }
}

// Pack 8 floats into a uint4 of halves.
__device__ __forceinline__ uint4 float_to_vec_f16(const float* in) {
    uint4 v;
    __half* h = reinterpret_cast<__half*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] = __float2half(in[i]);
    }
    return v;
}

// Unpack a uint4 containing 8 bf16s into a float array.
__device__ __forceinline__ void vec_to_float(uint4 v, float* out, const __nv_bfloat16* /*tag*/) {
    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __bfloat162float(h[i]);
    }
}

// Pack 8 floats into a uint4 of bf16s.
__device__ __forceinline__ uint4 float_to_vec_bf16(const float* in) {
    uint4 v;
    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] = __float2bfloat16(in[i]);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Type-dispatched pack/unpack (for generic template code)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ void unpack_vec(typename VecType<T>::Type v, float* out);

template <>
__device__ __forceinline__ void unpack_vec<float>(float4 v, float* out) {
    vec_to_float(v, out);
}

template <>
__device__ __forceinline__ void unpack_vec<__half>(uint4 v, float* out) {
    vec_to_float(v, out, (__half*)nullptr);
}

template <>
__device__ __forceinline__ void unpack_vec<__nv_bfloat16>(uint4 v, float* out) {
    vec_to_float(v, out, (__nv_bfloat16*)nullptr);
}

template <typename T>
__device__ __forceinline__ typename VecType<T>::Type pack_vec(const float* in);

template <>
__device__ __forceinline__ float4 pack_vec<float>(const float* in) {
    return float_to_vec_f32(in);
}

template <>
__device__ __forceinline__ uint4 pack_vec<__half>(const float* in) {
    return float_to_vec_f16(in);
}

template <>
__device__ __forceinline__ uint4 pack_vec<__nv_bfloat16>(const float* in) {
    return float_to_vec_bf16(in);
}
