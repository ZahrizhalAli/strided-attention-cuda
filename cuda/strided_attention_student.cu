#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <cmath> // For sqrtf and expf
#include <type_traits> // For std::is_same

#include <c10/cuda/CUDAException.h>

// CUDA kernel for the forward pass of strided attention
//
// T: data type (e.g., float, double, half)
//
// Computes: output = Softmax((Q * K_strided^T) / sqrt(head_dim)) * V_strided
//
// Grid/Block Dimensions:
// - gridDim.x: batch_size * num_heads
// - gridDim.y: seq_len (for queries)
// - blockDim.x: Should be a power of 2, e.g., 128, 256. Represents threads per query token.
//

#define MAX_STRIDED_SEQ_LEN 256

template<typename T>
struct acc_type {
    using type = float;
};
template<>
// struct acc_type<double> {
//     using type = float;
// };
struct acc_type<double> {
    using type = double;
};


template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int stride) {
    
    // =================================================================
    // TODO: IMPLEMENT YOUR STRIDED ATTENTION KERNEL HERE
    // =================================================================
    //
    // Hints:
    // 1. Calculate thread/block indices to figure out which output element
    //    this thread block is responsible for.
    //    - `blockIdx.x` can map to (batch, head).
    //    - `blockIdx.y` can map to the query token index (`i`).
    //    - `threadIdx.x` is the thread within the block.

    // 2. Load the query vector `q_i` for the current query token `i`.
    //    Since all threads in the block need it, you can load it into shared memory.

    // 3. Compute dot products `Q_i * K_j^T` for all `j` in the stride.
    //    - Parallelize the dot product calculation across threads in the block.
    //    - Each thread can compute a partial dot product and then you can use
    //      a parallel reduction (e.g., using shared memory and `__syncthreads()`)
    //      to get the final score.

    // 4. Compute Softmax.
    //    - Find the maximum score among the strided scores (for numerical stability). This is another parallel reduction.
    //    - Compute the exponential of each score (subtracting the max).
    //    - Compute the sum of the exponentials (another parallel reduction).
    //    - Normalize to get the final attention probabilities.

    // 5. Compute the weighted sum of value vectors `V_j`.
    //    - Use the attention probabilities to weight the `v_j` vectors.
    //    - This is another parallel dot-product-like operation. Each thread can
    //      compute a part of the final output vector component.

    // 6. Write the final output vector to `output_ptr`.

    using acc_t = typename acc_type<T>::type;

    // Example of getting indices (you'll need to expand on this):
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int query_idx = blockIdx.y; // This is 'i' in the outer loop

    const int strided_seq_len = (seq_len + stride - 1) / stride;

    if (strided_seq_len > MAX_STRIDED_SEQ_LEN || (head_dim % 4 != 0)) return;

    extern __shared__ char smem[];
    T* q_sh = (T*)smem;
    acc_t* scores_sh = (acc_t*)((char*)smem + head_dim * sizeof(T));
    acc_t* reduction_smem = (acc_t*)((char*)scores_sh + MAX_STRIDED_SEQ_LEN * sizeof(acc_t));

    long q_base_offset = (long)batch_idx * num_heads * seq_len * head_dim +
                         (long)head_idx * seq_len * head_dim +
                         (long)query_idx * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_sh[i] = q_ptr[q_base_offset + i];
    }
    __syncthreads();

    // compute Attention Scores (with conditional vectorization) ---
    const acc_t scale = 1.0 / sqrt((acc_t)head_dim);
    long k_base_offset = (long)batch_idx * num_heads * seq_len * head_dim +
                         (long)head_idx * seq_len * head_dim;

    for (int j_strided = threadIdx.x; j_strided < strided_seq_len; j_strided += blockDim.x) {
        const int j_actual = j_strided * stride;
        const long k_offset = k_base_offset + (long)j_actual * head_dim;
        
        acc_t score = 0.0;
        if constexpr (!std::is_same_v<T, double>) {
            // High-performance vectorized path for float and half
            const float4* q_vec = reinterpret_cast<const float4*>(q_sh);
            const float4* k_vec = reinterpret_cast<const float4*>(&k_ptr[k_offset]);
            #pragma unroll
            for (int k = 0; k < head_dim / 4; ++k) {
                float4 q_val = q_vec[k];
                float4 k_val = k_vec[k];
                score += q_val.x * k_val.x + q_val.y * k_val.y + q_val.z * k_val.z + q_val.w * k_val.w;
            }
        } else {
            // Correctness path for double precision
            for (int k = 0; k < head_dim; ++k) {
                score += (acc_t)q_sh[k] * (acc_t)k_ptr[k_offset + k];
            }
        }
        scores_sh[j_strided] = score * scale;
    }
    __syncthreads();

    // softmax 
    acc_t thread_max = -1e20;
    for (int i = threadIdx.x; i < strided_seq_len; i += blockDim.x) {
        thread_max = max(thread_max, scores_sh[i]);
    }
    reduction_smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { reduction_smem[threadIdx.x] = max(reduction_smem[threadIdx.x], reduction_smem[threadIdx.x + s]); }
        __syncthreads();
    }
    const acc_t max_score = reduction_smem[0];

    acc_t thread_sum = 0.0;
    for (int i = threadIdx.x; i < strided_seq_len; i += blockDim.x) {
        acc_t val = exp(scores_sh[i] - max_score);
        scores_sh[i] = val;
        thread_sum += val;
    }
    reduction_smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { reduction_smem[threadIdx.x] += reduction_smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const acc_t exp_sum = reduction_smem[0];

    const acc_t inv_sum = 1.0 / (exp_sum + 1e-6);
    for (int i = threadIdx.x; i < strided_seq_len; i += blockDim.x) {
        scores_sh[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted Sum of V 
    long v_base_offset = (long)batch_idx * num_heads * seq_len * head_dim +
                         (long)head_idx * seq_len * head_dim;
    long output_offset = q_base_offset;

    // Each thread handles one or more output dimensions (k_out)
    for (int k_out = threadIdx.x; k_out < head_dim; k_out += blockDim.x) {
        acc_t out_val = 0.0;
        // Each thread iterates through all strided values to accumulate its result.
        // Reads from v_ptr are coalesced across threads in a warp.
        for (int j_strided = 0; j_strided < strided_seq_len; ++j_strided) {
            const acc_t weight = scores_sh[j_strided];
            if (weight > 1e-9) {
                const int j_actual = j_strided * stride;
                const long v_offset = v_base_offset + (long)j_actual * head_dim;
                out_val += weight * (acc_t)v_ptr[v_offset + k_out];
            }
        }
        output_ptr[output_offset + k_out] = (T)out_val;
    }
}


// C++ function that dispatches the CUDA kernel
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);


    // Create an output tensor of the same shape as Q
    auto output = torch::empty_like(q);

    // Define grid and block dimensions
    // Grid: One block per (batch, head, query_token)
    dim3 gridDim(batch_size * num_heads, seq_len);
    // Block: Threads to parallelize the work for a single query token
    dim3 blockDim(256); // A common choice, can be tuned

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        // Correctly calculate shared memory size based on the specific data types
        using acc_t = typename acc_type<scalar_t>::type;
        // Shared memory reduced, as out_sh is no longer needed
        size_t shared_mem_size = (head_dim * sizeof(scalar_t)) +         // For q_sh
                                 (MAX_STRIDED_SEQ_LEN * sizeof(acc_t)) + // For scores_sh
                                 (256 * sizeof(acc_t));           // For reduction_smem

        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim, shared_mem_size>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            stride
        );
    }));

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}

