import torch
from torch.utils.cpp_extension import load
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CUDA kernel
# This will compile the C++/CUDA code and create a Python module
# The `verbose=True` flag is helpful for debugging compilation errors
strided_attention_lib = load(
    name="strided_attention_lib",
    sources=[os.path.join(current_dir, "strided_attention_student.cu")],
    verbose=True
)

def strided_attention_forward(q, k, v, stride):
    """
    Python wrapper for our custom strided_attention_forward CUDA kernel.
    """
    # PyTorch's extension loader automatically creates a Python binding
    # for the C++ function we registered with TORCH_EXTENSION_NAME.
    return strided_attention_lib.strided_attention_forward(q, k, v, stride)
