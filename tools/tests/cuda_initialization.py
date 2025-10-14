#!/usr/bin/env python3
"""
CUDA and cuDNN Diagnostics Script

This script checks CUDA and cuDNN environment setup for PyTorch training.
It verifies hardware availability, configuration, and tests basic operations.

Usage:
    python tools/tests/cuda_initialization.py
"""

import sys
import traceback
import torch
import torch.nn as nn

def check_cuda_basic():
    """Check basic CUDA availability and device info."""
    print("=== Basic CUDA Check ===")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot continue cuDNN testing")
        return False

    print(f"device count: {torch.cuda.device_count()}")
    print(f"current device: {torch.cuda.current_device()}")
    print(f"device name: {torch.cuda.get_device_name()}")

    # Check GPU memory
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024**3
    print(f"GPU memory: {total_memory:.2f} GB")

    return True

def check_cudnn():
    """Check cuDNN availability and configuration."""
    print("\n=== cuDNN Check ===")

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping cuDNN check")
        return False

    print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    if not torch.backends.cudnn.is_available():
        print("❌ cuDNN not available - this may cause training failures")
        return False

    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    # Check TF32 settings (important for RTX 5090 performance)
    print(f"cuDNN allow_tf32: {torch.backends.cudnn.allow_tf32}")
    print(f"CUDA matmul allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")

    if torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32:
        print("✅ TF32 enabled for optimal RTX 5090 performance")
    else:
        print("⚠️  TF32 disabled - may impact performance on RTX 5090")

    return True

def test_cudnn_operations():
    """Test basic cuDNN operations."""
    print("\n=== cuDNN Operations Test ===")

    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        print("❌ cuDNN not available - skipping operations test")
        return False

    device = torch.device('cuda')

    try:
        # Test 1: Simple convolution
        print("Testing 2D convolution...")
        input_tensor = torch.randn(1, 3, 224, 224, device=device)
        conv = nn.Conv2d(3, 64, 3, padding=1).to(device)

        with torch.no_grad():
            output = conv(input_tensor)
            print(f"  Input: {input_tensor.shape} -> Output: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("✅ 2D convolution successful")

        # Test 2: BatchNorm (uses cuDNN internally)
        print("Testing Conv2D + BatchNorm...")
        conv_bn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ).to(device)

        with torch.no_grad():
            output = conv_bn(input_tensor)
            print(f"  Input: {input_tensor.shape} -> Output: {output.shape}")
        print("✅ Conv2D + BatchNorm successful")

        # Test 3: Multiple layers (stress test)
        print("Testing sequential convolutions...")
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1)
        ).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            print(f"  Input: {input_tensor.shape} -> Output: {output.shape}")
        print("✅ Sequential convolutions successful")

        return True

    except Exception as e:
        print(f"❌ cuDNN operation failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_memory_allocation():
    """Test cuDNN memory allocation and workspace."""
    print("\n=== Memory Allocation Test ===")

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping memory test")
        return False

    device = torch.device('cuda')

    try:
        # Test larger operations that require cuDNN workspace
        print("Testing cuDNN workspace allocation...")

        # Large convolution
        large_input = torch.randn(2, 64, 256, 256, device=device)
        large_conv = nn.Conv2d(64, 128, 7, padding=3).to(device)

        with torch.no_grad():
            output = large_conv(large_input)
            print(f"  Large conv: {large_input.shape} -> {output.shape}")

        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"  Memory usage: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

        print("✅ Memory allocation test successful")
        return True

    except Exception as e:
        print(f"❌ Memory allocation failed: {e}")
        return False

def apply_cudnn_optimizations():
    """Apply cuDNN optimizations for RTX 5090."""
    print("\n=== cuDNN Optimizations ===")

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping optimizations")
        return

    # Enable TF32 for performance (already enabled in main training script)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Enable benchmark mode for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("✅ cuDNN optimizations applied:")
    print(f"  TF32 enabled: {torch.backends.cudnn.allow_tf32}")
    print(f"  Benchmark mode: {torch.backends.cudnn.benchmark}")
    print(f"  Deterministic: {torch.backends.cudnn.deterministic}")

def main():
    """Main diagnostic function."""
    print("CUDA and cuDNN Diagnostics")
    print("=" * 40)

    success = True

    # Basic CUDA check
    if not check_cuda_basic():
        return 1

    # cuDNN check
    if not check_cudnn():
        success = False

    # Apply optimizations
    apply_cudnn_optimizations()

    # Test cuDNN operations
    if not test_cudnn_operations():
        success = False

    # Test memory allocation
    if not test_memory_allocation():
        success = False

    print("\n" + "=" * 40)
    if success:
        print("✅ All diagnostics passed - your system is ready for cuDNN training!")
    else:
        print("⚠️  Some tests failed - please check the errors above")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
