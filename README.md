# Custom CUDA Attention Kernel for a Transformer

## Project Goal

The goal is to replace a standard, naive PyTorch attention implementation with a high-performance version written in CUDA. This will involve:
1.  Understanding the mathematics of Strided Attention.
2.  Designing and implementing a CUDA kernel to perform this operation efficiently.
3.  Using PyTorch's C++ extension functionality to compile and link your CUDA code.
4.  Integrating your custom kernel into a simplified Transformer model.
5.  Testing for correctness and benchmarking performance.

## File Structure

Overview of the skeleton code provided, inside `src/`:

* `model.py`: Contains a simplified Transformer-like model. No need to edit this file, but, though understand how it uses the attention layer.
* `attention_student.py`: This is where the main logic for the attention mechanism resides.
    * `naive_strided_attention`: A reference implementation of strided attention in pure PyTorch. Use this to verify the correctness of your CUDA kernel.
    * `CustomStridedAttention`: A class where it will call the custom CUDA kernel. The `forward` method located here.
* `test.py`: A testing suite.
    * It verifies that the CUDA implementation produces the same output as the naive PyTorch version.
    * It includes a simple benchmark to compare the performance.
* `cuda/`: This directory contains all CUDA-related code.
    * `strided_attention_student.cu`: It contains the skeleton for the CUDA kernel. 
    * `binding.py`: This file handles the bridge between PyTorch and the CUDA code. It uses `torch.utils.cpp_extension` to load, compile, and create a Python binding for your C++/CUDA functions. 

## Strided Attention

Standard attention allows every token in a sequence to attend to every other token. Strided attention is a sparse attention pattern where each query token attends only to key tokens at a fixed `stride`.

For example, with a stride of 2, query `q_i` attends to keys `k_0, k_2, k_4, ...`.

## Preparation

```
conda create --name [env-name] python=3.10.12
conda activate [env-name]
pip install torch numpy
```

## How to Run


When first time running the `test.py` script, PyTorch's C++ extension will automatically compile the CUDA code. This may take a minute. Subsequent runs will be much faster as it will use the cached compiled library.

Perlmutter Interactive Session:
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account m4999
# Correctness Test
python test.py --mode=correctness

# Benchmark performance
python test.py --mode=benchmark
```