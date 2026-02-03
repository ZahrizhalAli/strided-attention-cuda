import torch
import time
import argparse
from model import SimpleModel
from attention_student import *#naive_strided_attention

def test_correctness(device):
    print("--- Running Correctness Test ---")
    
    batch_size = 1#2
    seq_len = 2#16
    embed_dim = 32
    num_heads = 1#4
    stride = 1#2
    
    torch.manual_seed(0)
    # Use float64 for high precision checking
    model = SimpleModel(embed_dim, num_heads, stride).to(device).double()
    model.eval()

    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.double)
    
    # Get the Q, K, V projections to test the attention module directly
    qkv = model.layer.attention.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, model.layer.attention.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # 1. Run the naive PyTorch implementation
    print("Running naive PyTorch implementation...")
    naive_output = naive_strided_attention(q, k, v, stride)
    
    # 2. Run the custom CUDA kernel implementation
    print("Running custom CUDA implementation...")
    custom_attn = CustomStridedAttention(embed_dim, num_heads, stride).to(device).double()
    # Manually set the weights to be the same as the model's projection layer
    custom_attn.qkv_proj.weight.data = model.layer.attention.qkv_proj.weight.data.clone()
    custom_attn.qkv_proj.bias.data = model.layer.attention.qkv_proj.bias.data.clone()
    
    # We need to get the output from just the attention calculation, not the final projection
    # So we call the kernel directly via the binding
    from cuda.binding import strided_attention_forward
    custom_output = strided_attention_forward(q.contiguous(), k.contiguous(), v.contiguous(), stride)

    # 3. Compare the results
    print("Comparing results...")
    is_close = torch.allclose(naive_output, custom_output, atol=1e-6)
    
    if is_close:
        print("\nCorrectness Test Passed!")
    else:
        print("\nCorrectness Test Failed!")
        print("Max difference:", (naive_output - custom_output).abs().max().item())

    print("-" * 30)


def test_benchmark(device):
    print("--- Running Benchmark Test ---")
    
    batch_size = 16
    seq_len = 1024
    embed_dim = 512
    num_heads = 8
    stride = 4
    
    warmup_iter = 10
    bench_iter = 100

    model = SimpleModel(embed_dim, num_heads, stride).to(device).float()
    model.eval()

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float)
    
    qkv = model.layer.attention.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, model.layer.attention.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
    
    # --- Benchmark Naive PyTorch ---
    print("Warming up naive implementation...")
    for _ in range(warmup_iter):
        _ = naive_strided_attention(q, k, v, stride)
    torch.cuda.synchronize()
    
    print("Running benchmark for naive implementation...")
    start_time = time.time()
    for _ in range(bench_iter):
        _ = naive_strided_attention(q, k, v, stride)
    torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / bench_iter
    print(f"Naive PyTorch implementation average time: {naive_time * 1000:.4f} ms")
    
    # --- Benchmark Custom CUDA ---
    from cuda.binding import strided_attention_forward
    print("\nWarming up custom CUDA implementation...")
    for _ in range(warmup_iter):
        _ = strided_attention_forward(q, k, v, stride)
    torch.cuda.synchronize()

    print("Running benchmark for custom CUDA implementation...")
    start_time = time.time()
    for _ in range(bench_iter):
        _ = strided_attention_forward(q, k, v, stride)
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / bench_iter
    print(f"Custom CUDA implementation average time: {custom_time * 1000:.4f} ms")

    # --- Results ---
    speedup = naive_time / custom_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "benchmark"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This assignment requires a CUDA-enabled GPU.")
        exit()
    
    device = torch.device("cuda")
    
    if args.mode == "correctness":
        test_correctness(device)
    elif args.mode == "benchmark":
        test_benchmark(device)