#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>

constexpr int NS = 16;
constexpr int TC = 32;

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T, int CPB, int S>
__global__ void __launch_bounds__(CPB * S)
scan_kernel(const T* __restrict__ x, const T* __restrict__ dt, const float* __restrict__ A,
            const T* __restrict__ Bm, const T* __restrict__ Cm, const float* __restrict__ D,
            T* __restrict__ y, int L, int E) {
    constexpr int NP = NS / S;
    constexpr int NT = CPB * S;
    __shared__ float sB[TC * NS];
    __shared__ float sC[TC * NS];

    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int cl = tid / S;
    const int sub = tid % S;
    const int e = blockIdx.x * CPB + cl;
    const bool valid = e < E;

    float a[NP];
    float h[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        a[i] = valid ? A[e * NS + sub * NP + i] : 0.f;
        h[i] = 0.f;
    }
    const float Dv = valid ? D[e] : 0.f;

    const size_t row0 = (size_t)b * L;
    const T* xp = x + row0 * E + (valid ? e : 0);
    const T* dp = dt + row0 * E + (valid ? e : 0);
    T* yp = y + row0 * E + (valid ? e : 0);
    const T* Bp = Bm + row0 * NS;
    const T* Cp = Cm + row0 * NS;

    for (int t0 = 0; t0 < L; t0 += TC) {
        const int tl = min(TC, L - t0);
        for (int idx = tid; idx < tl * NS; idx += NT) {
            sB[idx] = to_f(Bp[(size_t)t0 * NS + idx]);
            sC[idx] = to_f(Cp[(size_t)t0 * NS + idx]);
        }
        __syncthreads();
#pragma unroll 8
        for (int t = 0; t < tl; ++t) {
            const size_t off = (size_t)(t0 + t) * E;
            const float xv = valid ? to_f(xp[off]) : 0.f;
            const float dv = valid ? to_f(dp[off]) : 0.f;
            const float dx = dv * xv;
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const float ev = __expf(dv * a[i]);
                h[i] = ev * h[i] + dx * sB[t * NS + sub * NP + i];
                part += h[i] * sC[t * NS + sub * NP + i];
            }
#pragma unroll
            for (int o = S / 2; o >= 1; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            if (valid && sub == 0) st(yp + off, part + Dv * xv);
        }
        __syncthreads();
    }
}

template <typename U>
struct Tag { using type = U; };

template <typename F>
void dispatch(bool bf16, int cpb, int s, F&& f) {
#define DS(T, CPB, S) if (s == S) { f(Tag<T>{}, std::integral_constant<int, CPB>{}, std::integral_constant<int, S>{}); return; }
#define DC(T, CPB) if (cpb == CPB) { DS(T, CPB, 1) DS(T, CPB, 2) DS(T, CPB, 4) DS(T, CPB, 8) DS(T, CPB, 16) }
#define DT(T) { DC(T, 16) DC(T, 32) DC(T, 64) DC(T, 128) DC(T, 256) DC(T, 512) }
    if (bf16) DT(__nv_bfloat16) else DT(float)
#undef DT
#undef DC
#undef DS
    TORCH_CHECK(false, "unsupported (cpb, s) = (", cpb, ", ", s, ")");
}

torch::Tensor scan(torch::Tensor x, torch::Tensor dt, torch::Tensor A, torch::Tensor Bm,
                   torch::Tensor Cm, torch::Tensor D, int64_t cpb, int64_t s) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && dt.is_contiguous() && Bm.is_contiguous() && Cm.is_contiguous(), "contiguous cuda tensors required");
    TORCH_CHECK(x.dim() == 3 && Bm.size(2) == NS && A.size(1) == NS, "d_state must be 16");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && D.scalar_type() == torch::kFloat32, "A and D must be fp32");
    TORCH_CHECK(x.scalar_type() == dt.scalar_type() && x.scalar_type() == Bm.scalar_type() && x.scalar_type() == Cm.scalar_type(), "x, delta, B, C must share a dtype");
    TORCH_CHECK(cpb * s <= 1024 && cpb * s >= 32, "threads per block must be in [32, 1024]");
    const bool bf16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(bf16 || x.scalar_type() == torch::kFloat32, "fp32 or bf16 only");
    const c10::cuda::CUDAGuard guard(x.device());
    const int Bsz = x.size(0), L = x.size(1), E = x.size(2);
    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto Ac = A.contiguous();
    auto Dc = D.contiguous();
    dispatch(bf16, (int)cpb, (int)s, [&](auto tag, auto cpb_c, auto s_c) {
        using T = typename decltype(tag)::type;
        constexpr int CPB = decltype(cpb_c)::value;
        constexpr int S = decltype(s_c)::value;
        dim3 grid((E + CPB - 1) / CPB, Bsz);
        scan_kernel<T, CPB, S><<<grid, CPB * S, 0, stream>>>(
            reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(dt.data_ptr()),
            Ac.data_ptr<float>(), reinterpret_cast<const T*>(Bm.data_ptr()),
            reinterpret_cast<const T*>(Cm.data_ptr()), Dc.data_ptr<float>(),
            reinterpret_cast<T*>(y.data_ptr()), L, E);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<int64_t> kernel_info(bool bf16, int64_t cpb, int64_t s) {
    std::vector<int64_t> out;
    TORCH_CHECK(cpb * s <= 1024 && cpb * s >= 32, "threads per block must be in [32, 1024]");
    dispatch(bf16, (int)cpb, (int)s, [&](auto tag, auto cpb_c, auto s_c) {
        using T = typename decltype(tag)::type;
        constexpr int CPB = decltype(cpb_c)::value;
        constexpr int S = decltype(s_c)::value;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, scan_kernel<T, CPB, S>);
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scan_kernel<T, CPB, S>, CPB * S, 0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        out = {attr.numRegs, (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes, blocks,
               (int64_t)blocks * CPB * S, prop.maxThreadsPerMultiProcessor, prop.multiProcessorCount};
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &scan, "selective scan");
    m.def("kernel_info", &kernel_info, "regs, smem, local, blocks/SM, threads/SM, max threads/SM, SMs");
}
