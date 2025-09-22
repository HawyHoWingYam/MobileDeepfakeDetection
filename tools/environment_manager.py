#!/usr/bin/env python3
"""
AWARE-NET Environment Manager
Automatic PyTorch installation based on GPU compatibility

This script detects GPU architecture and installs the appropriate PyTorch version:
- RTX 50-series (sm_120): PyTorch nightly with CUDA 12.6+
- RTX 30/40-series (sm_86/89): PyTorch stable with CUDA 12.4
- CPU-only: PyTorch stable CPU version
"""

import os
import sys
import subprocess
import platform
import argparse
from typing import Dict, Optional, Tuple, List

def detect_gpu_architecture() -> Tuple[Optional[str], Optional[int]]:
    """
    Detect GPU architecture using nvidia-smi

    Returns:
        Tuple of (gpu_name, sm_version) or (None, None) if no GPU
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0].strip():
            return None, None

        gpu_info = lines[0].split(', ')
        if len(gpu_info) >= 2:
            gpu_name = gpu_info[0].strip()
            compute_cap = float(gpu_info[1].strip())
            sm_version = int(compute_cap * 10)  # 8.6 -> 86, 12.0 -> 120

            return gpu_name, sm_version

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None, None

    return None, None

def get_pytorch_install_command(gpu_name: Optional[str], sm_version: Optional[int]) -> Dict[str, str]:
    """
    Get appropriate PyTorch installation command based on GPU

    Returns:
        Dictionary with installation commands and rationale
    """
    if gpu_name is None or sm_version is None:
        return {
            'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
            'type': 'cpu',
            'rationale': 'No CUDA-capable GPU detected, installing CPU version'
        }

    # RTX 50 series detection
    if sm_version >= 120:  # sm_120 and above
        return {
            'command': 'pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126',
            'type': 'cuda_nightly',
            'rationale': f'RTX 50-series GPU detected ({gpu_name}, sm_{sm_version}), using PyTorch nightly with CUDA 12.6'
        }

    # RTX 30/40 series and other modern GPUs
    elif sm_version >= 75:  # RTX 2060 and above
        return {
            'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124',
            'type': 'cuda_stable',
            'rationale': f'Modern GPU detected ({gpu_name}, sm_{sm_version}), using PyTorch stable with CUDA 12.4'
        }

    # Older GPUs
    else:
        return {
            'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
            'type': 'cuda_legacy',
            'rationale': f'Older GPU detected ({gpu_name}, sm_{sm_version}), using PyTorch with CUDA 11.8'
        }

def install_pytorch(dry_run: bool = False) -> bool:
    """
    Install PyTorch with appropriate CUDA version

    Args:
        dry_run: If True, only show what would be installed

    Returns:
        True if successful, False otherwise
    """
    print("🔍 Detecting GPU architecture...")
    gpu_name, sm_version = detect_gpu_architecture()

    if gpu_name:
        print(f"   GPU detected: {gpu_name}")
        print(f"   Compute capability: sm_{sm_version}")
    else:
        print("   No CUDA-capable GPU detected")

    install_info = get_pytorch_install_command(gpu_name, sm_version)

    print(f"\\n📦 PyTorch installation plan:")
    print(f"   Type: {install_info['type']}")
    print(f"   Rationale: {install_info['rationale']}")
    print(f"   Command: {install_info['command']}")

    if dry_run:
        print("\\n🔍 Dry run mode - no installation performed")
        return True

    print("\\n⚡ Installing PyTorch...")
    try:
        result = subprocess.run(install_info['command'].split(), check=True)
        print("✅ PyTorch installation completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False

def verify_installation() -> bool:
    """
    Verify PyTorch installation and CUDA availability

    Returns:
        True if verification successful
    """
    print("\\n🧪 Verifying PyTorch installation...")

    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   CUDA available: Yes")
            print(f"   GPU count: {gpu_count}")
            print(f"   Primary GPU: {gpu_name}")

            # Test basic operations
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.t())
            print(f"   CUDA operations: ✅ Working")

        else:
            print(f"   CUDA available: No (CPU-only mode)")

        return True

    except ImportError:
        print("   ❌ PyTorch not found after installation")
        return False
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AWARE-NET Environment Manager")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show installation plan without executing')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing installation')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU-only installation')

    args = parser.parse_args()

    print("🚀 AWARE-NET Environment Manager")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")

    if args.verify_only:
        success = verify_installation()
        sys.exit(0 if success else 1)

    # Override GPU detection if force-cpu
    if args.force_cpu:
        print("\\n⚠️  CPU-only mode forced")
        gpu_name, sm_version = None, None

    # Install PyTorch
    success = install_pytorch(dry_run=args.dry_run)

    if success and not args.dry_run:
        # Verify installation
        verify_success = verify_installation()
        if not verify_success:
            print("\\n⚠️  Installation completed but verification failed")
            sys.exit(1)

    print("\\n🎉 Environment setup completed!")
    if not args.dry_run:
        print("\\nNext steps:")
        print("  1. conda install remaining dependencies from environment.yml")
        print("  2. pip install -e . to install AWARE-NET in development mode")
        print("  3. Test with: python -c \\"import torch; print(torch.cuda.is_available())\\"")

if __name__ == '__main__':
    main()