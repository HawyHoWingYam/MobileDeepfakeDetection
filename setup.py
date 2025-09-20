#!/usr/bin/env python3
"""
AWARE-NET Master Setup Script
Complete environment setup, dataset preparation, and validation in one script
"""

import os
import sys
import json
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class AwareNetMasterSetup:
    """All-in-one AWARE-NET setup and management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.system = platform.system()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Add src to path for imports
        sys.path.append(str(self.project_root / "src"))
        
    # =========================
    # GPU & PyTorch Management  
    # =========================
    
    def detect_gpu_architecture(self) -> Optional[Dict]:
        """Detect GPU and determine optimal PyTorch installation"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'count': torch.cuda.device_count()
            }
            
            # Determine architecture category
            major, minor = gpu_info['capability']
            arch_code = major * 10 + minor
            
            if arch_code >= 120:  # RTX 50 series
                gpu_info['category'] = 'rtx50'
                gpu_info['pytorch_strategy'] = 'nightly_cu129'
            elif arch_code >= 86:  # RTX 30/40 series
                gpu_info['category'] = 'rtx30_40'
                gpu_info['pytorch_strategy'] = 'stable_cu124'
            elif arch_code >= 75:  # RTX 20 series
                gpu_info['category'] = 'rtx20'
                gpu_info['pytorch_strategy'] = 'stable_cu121'
            else:  # Older GPUs
                gpu_info['category'] = 'legacy'
                gpu_info['pytorch_strategy'] = 'stable_cpu'
                
            return gpu_info
            
        except ImportError:
            return None
        except Exception as e:
            print(f"GPU detection error: {e}")
            return None
    
    def install_pytorch_optimized(self, strategy: str = None, force_clean: bool = False) -> bool:
        """Install PyTorch optimized for detected GPU with comprehensive cleanup"""
        
        if strategy is None:
            gpu_info = self.detect_gpu_architecture()
            strategy = gpu_info['pytorch_strategy'] if gpu_info else 'stable_cpu'
        
        print(f"🚀 Installing PyTorch with strategy: {strategy}")
        
        # Step 1: Comprehensive cleanup if needed
        if force_clean:
            print("🧹 Performing comprehensive cleanup...")
            cleanup_commands = [
                "python -m pip uninstall torch torchvision torchaudio torchmetrics -y",
                "python -m pip cache purge"
            ]
            for cmd in cleanup_commands:
                self._run_command(cmd, f"Cleanup: {cmd}", critical=False)
        
        # Installation commands for different strategies
        commands = {
            'nightly_cu129': [
                "python -m pip uninstall torch torchvision torchaudio -y",
                "python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129"
            ],
            'stable_cu124': [
                "python -m pip uninstall torch torchvision torchaudio -y",
                "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
            ],
            'stable_cu121': [
                "python -m pip uninstall torch torchvision torchaudio -y", 
                "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            ],
            'stable_cpu': [
                "python -m pip uninstall torch torchvision torchaudio -y",
                "python -m pip install torch torchvision torchaudio"
            ]
        }
        
        if strategy not in commands:
            print(f"❌ Unknown strategy: {strategy}")
            return False
        
        # Execute installation commands
        for cmd in commands[strategy]:
            success = self._run_command(cmd, f"Executing: {cmd}")
            if not success:
                print(f"❌ Failed at command: {cmd}")
                # Try fallback strategies
                return self._try_fallback_installation(strategy)
        
        # Verify installation
        return self._verify_pytorch_installation()
    
    def _try_fallback_installation(self, failed_strategy: str) -> bool:
        """Try fallback PyTorch installation strategies"""
        
        print("🔄 Trying fallback installation strategies...")
        
        # Define fallback chain
        fallback_chain = {
            'nightly_cu129': ['stable_cu124', 'stable_cu121', 'stable_cpu'],
            'stable_cu124': ['stable_cu121', 'stable_cpu'],
            'stable_cu121': ['stable_cpu'],
            'stable_cpu': []
        }
        
        fallbacks = fallback_chain.get(failed_strategy, [])
        
        for fallback_strategy in fallbacks:
            print(f"🔄 Trying fallback: {fallback_strategy}")
            
            if fallback_strategy == 'stable_cu124':
                cmd = "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
            elif fallback_strategy == 'stable_cu121':
                cmd = "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            elif fallback_strategy == 'stable_cpu':
                cmd = "python -m pip install torch torchvision torchaudio"
            else:
                continue
            
            success = self._run_command(cmd, f"Fallback installation: {fallback_strategy}")
            if success and self._verify_pytorch_installation():
                print(f"✅ Fallback {fallback_strategy} succeeded!")
                return True
        
        print("❌ All fallback strategies failed")
        return False
    
    def fix_pytorch_installation(self) -> bool:
        """Fix broken PyTorch installation with comprehensive cleanup"""
        
        print("🔧 Fixing PyTorch Installation Issues")
        print("=" * 50)
        
        # Force clean installation
        return self.install_pytorch_optimized(force_clean=True)
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies"""
        
        print("📦 Installing AWARE-NET dependencies...")
        
        # Core dependencies
        core_packages = [
            "torchmetrics>=1.0.0",
            "timm>=1.0.19",
            "albumentations>=1.3.1", 
            "opencv-python>=4.8.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0"
        ]
        
        # Optional but recommended packages
        optional_packages = [
            "jupyter",
            "notebook", 
            "ipykernel",
            "wandb",
            "tensorboard",
            "onnx",
            "onnxruntime"
        ]
        
        # Install core packages
        for package in core_packages:
            cmd = f"python -m pip install {package}"
            success = self._run_command(cmd, f"Installing {package}")
            if not success:
                print(f"⚠️  Failed to install {package} (continuing...)")
        
        # Install optional packages (failures are non-critical)
        for package in optional_packages:
            cmd = f"python -m pip install {package}"
            self._run_command(cmd, f"Installing optional: {package}", critical=False)
        
        return True
    
    def _verify_pytorch_installation(self) -> bool:
        """Verify PyTorch installation works"""
        
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} imported successfully")
            
            # Test basic functionality
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x * 2
            print("✅ PyTorch tensor operations work")
            
            # Test CUDA if available
            if torch.cuda.is_available():
                try:
                    x_cuda = x.cuda()
                    print("✅ CUDA operations work")
                except Exception as e:
                    print(f"⚠️  CUDA available but not functional: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ PyTorch verification failed: {e}")
            return False
    
    # =========================
    # Dataset Management
    # =========================
    
    def prepare_dataset(self) -> bool:
        """Prepare CelebDF-v2 dataset with manifest generation"""
        
        print("📊 CelebDF-v2 Dataset Preparation")
        print("=" * 50)
        
        # Check configuration
        config_path = self.project_root / "configs" / "dataset_paths.json"
        if not config_path.exists():
            print("❌ Configuration file not found: configs/dataset_paths.json")
            return False
        
        # Load configuration
        try:
            from utils.dataset_config import DatasetConfig
            from utils.manifest_generator import ManifestGenerator
            
            config = DatasetConfig(config_path)
            
            # Get dataset paths
            real_dir = Path(config.config["paths"]["real_images"])
            fake_dir = Path(config.config["paths"]["fake_images"])
            
            print(f"📁 Real images: {real_dir}")
            print(f"📁 Fake images: {fake_dir}")
            
            # Check if directories exist
            if not real_dir.exists():
                print(f"❌ Real images directory not found: {real_dir}")
                return False
            
            if not fake_dir.exists():
                print(f"❌ Fake images directory not found: {fake_dir}")
                return False
            
            # Count images
            print("\\n🔍 Scanning for images...")
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            
            real_images = []
            for ext in extensions:
                real_images.extend(list(real_dir.glob(f"**/{ext}")))
            
            fake_images = []
            for ext in extensions:
                fake_images.extend(list(fake_dir.glob(f"**/{ext}")))
            
            print(f"Found {len(real_images)} real images")
            print(f"Found {len(fake_images)} fake images")
            print(f"Total: {len(real_images) + len(fake_images)} images")
            
            if len(real_images) == 0 or len(fake_images) == 0:
                print("❌ No images found in one or both directories")
                return False
            
            # Create manifest generator
            print("\\n🛠️ Creating manifest files...")
            generator = ManifestGenerator(
                config=config,
                validate_images=False,  # Skip validation for speed
                calculate_md5=False,    # Skip MD5 for speed
                seed=42
            )
            
            # Generate manifests
            manifest_paths = generator.generate_full_dataset_manifest(
                real_dir=real_dir,
                fake_dir=fake_dir
            )
            
            print("\\n✅ Manifest generation completed!")
            for split_name, manifest_path in manifest_paths.items():
                print(f"{split_name}: {manifest_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset preparation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================
    # Environment Validation
    # =========================
    
    def validate_environment(self) -> Dict[str, Any]:
        """Comprehensive environment validation"""
        
        print("🔍 AWARE-NET Environment Validation")
        print("=" * 50)
        
        results = {
            'python': self._validate_python(),
            'pytorch': self._validate_pytorch(),
            'gpu': self._validate_gpu(),
            'dependencies': self._validate_dependencies(),
            'project': self._validate_project_structure()
        }
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _validate_python(self) -> Dict[str, Any]:
        """Validate Python environment"""
        
        python_info = {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro
            },
            'executable': sys.executable,
        }
        
        # Check Python version
        min_version = (3, 8)
        current_version = (sys.version_info.major, sys.version_info.minor)
        python_info['version_ok'] = current_version >= min_version
        
        if python_info['version_ok']:
            print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        else:
            print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (requires >= 3.8)")
        
        return python_info
    
    def _validate_pytorch(self) -> Dict[str, Any]:
        """Validate PyTorch installation"""
        
        pytorch_info = {
            'installed': False,
            'version': None,
            'cuda_available': False,
            'basic_ops': False
        }
        
        try:
            import torch
            pytorch_info['installed'] = True
            pytorch_info['version'] = torch.__version__
            pytorch_info['cuda_available'] = torch.cuda.is_available()
            
            print(f"✅ PyTorch {torch.__version__}")
            
            if pytorch_info['cuda_available']:
                print(f"✅ CUDA {torch.version.cuda}")
            else:
                print("⚠️  CUDA not available")
            
            # Test basic operations
            try:
                x = torch.tensor([1.0, 2.0, 3.0])
                y = x * 2
                assert y.sum().item() == 12.0
                pytorch_info['basic_ops'] = True
                print("✅ Basic tensor operations")
            except Exception as e:
                pytorch_info['basic_ops'] = False
                print(f"❌ Basic operations failed: {e}")
            
        except ImportError:
            pytorch_info['installed'] = False
            print("❌ PyTorch not installed")
        except Exception as e:
            pytorch_info['error'] = str(e)
            print(f"❌ PyTorch validation error: {e}")
        
        return pytorch_info
    
    def _validate_gpu(self) -> Dict[str, Any]:
        """Validate GPU functionality"""
        
        gpu_info = {
            'available': False,
            'devices': [],
            'functional': False
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                
                # Get information for each GPU
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'capability': torch.cuda.get_device_capability(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    }
                    
                    # Test GPU functionality
                    try:
                        torch.cuda.set_device(i)
                        test_tensor = torch.zeros(100).cuda()
                        test_result = test_tensor.sum()
                        device_info['functional'] = True
                        print(f"✅ GPU {i}: {device_info['name']} (functional)")
                    except Exception as e:
                        device_info['functional'] = False
                        device_info['error'] = str(e)
                        print(f"❌ GPU {i}: {device_info['name']} (error: {str(e)[:50]})")
                    
                    gpu_info['devices'].append(device_info)
                
                # Overall GPU functionality
                gpu_info['functional'] = any(dev['functional'] for dev in gpu_info['devices'])
                
            else:
                print("⚠️  No CUDA-capable GPU detected")
                
        except ImportError:
            print("❌ Cannot validate GPU (PyTorch not available)")
        except Exception as e:
            gpu_info['error'] = str(e)
            print(f"❌ GPU validation error: {e}")
        
        return gpu_info
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies"""
        
        # Core dependencies required for AWARE-NET
        required_packages = {
            'torchmetrics': 'PyTorch metrics',
            'timm': 'Pre-trained models',
            'albumentations': 'Data augmentation',
            'opencv-python': 'Computer vision',
            'pandas': 'Data processing',
            'numpy': 'Numerical computing',
            'scikit-learn': 'Machine learning',
            'matplotlib': 'Plotting',
            'seaborn': 'Statistics plotting',
            'tqdm': 'Progress bars'
        }
        
        deps_info = {
            'required': {},
            'missing_required': []
        }
        
        # Check required packages
        for package, description in required_packages.items():
            try:
                if package == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                elif package == 'scikit-learn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                
                deps_info['required'][package] = {
                    'installed': True,
                    'version': version,
                    'description': description
                }
                print(f"✅ {package} ({version})")
                
            except ImportError:
                deps_info['required'][package] = {
                    'installed': False,
                    'description': description
                }
                deps_info['missing_required'].append(package)
                print(f"❌ {package} - {description}")
        
        return deps_info
    
    def _validate_project_structure(self) -> Dict[str, Any]:
        """Validate AWARE-NET project structure"""
        
        required_files = [
            'src/stage_00/train_baseline.py',
            'src/stage_00/dataset.py',
            'src/stage_00/baseline_model.py',
            'configs/dataset_paths.json',
            'configs/training_config.json',
            'train.py'
        ]
        
        project_info = {
            'files': {},
            'manifests': {},
            'missing_files': []
        }
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            project_info['files'][file_path] = exists
            
            if exists:
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
                project_info['missing_files'].append(file_path)
        
        # Check manifest files
        manifests_dir = self.project_root / "manifests"
        if manifests_dir.exists():
            manifest_files = list(manifests_dir.glob("*.csv"))
            project_info['manifests'] = {
                'directory_exists': True,
                'files': [f.name for f in manifest_files],
                'count': len(manifest_files)
            }
            
            if manifest_files:
                print(f"✅ Found {len(manifest_files)} manifest files")
            else:
                print("⚠️  Manifests directory exists but no CSV files found")
        else:
            project_info['manifests'] = {
                'directory_exists': False,
                'files': [],
                'count': 0
            }
            print("❌ Manifests directory not found")
        
        return project_info
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        
        summary = {
            'overall_status': 'unknown',
            'critical_issues': [],
            'recommendations': []
        }
        
        # Check critical components
        python_ok = results['python'].get('version_ok', False)
        pytorch_ok = results['pytorch'].get('installed', False)
        deps_ok = len(results['dependencies'].get('missing_required', [])) == 0
        project_ok = len(results['project'].get('missing_files', [])) == 0
        
        critical_checks = [python_ok, pytorch_ok, deps_ok, project_ok]
        
        if all(critical_checks):
            summary['overall_status'] = 'excellent'
        elif python_ok and pytorch_ok and len(results['dependencies'].get('missing_required', [])) <= 2:
            summary['overall_status'] = 'good'
        elif python_ok and pytorch_ok:
            summary['overall_status'] = 'fair'
        else:
            summary['overall_status'] = 'poor'
        
        # Generate recommendations
        if not python_ok:
            summary['critical_issues'].append("Python version too old (requires >= 3.8)")
            summary['recommendations'].append("Upgrade Python to version 3.8 or higher")
        
        if not pytorch_ok:
            summary['critical_issues'].append("PyTorch not installed or not working")
            summary['recommendations'].append("Run this script with option 2 (Install PyTorch)")
        
        missing_deps = results['dependencies'].get('missing_required', [])
        if missing_deps:
            summary['critical_issues'].append(f"Missing required packages: {', '.join(missing_deps)}")
            summary['recommendations'].append("Run this script with option 3 (Install Dependencies)")
        
        if not results['project']['manifests'].get('files'):
            summary['recommendations'].append("Run this script with option 4 (Prepare Dataset)")
        
        return summary
    
    # =========================
    # Utility Methods
    # =========================
    
    def _run_command(self, cmd: str, description: str = "", critical: bool = True) -> bool:
        """Execute command with error handling"""
        
        if description:
            print(f"🔄 {description}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            return True
            
        except subprocess.CalledProcessError as e:
            if critical:
                print(f"❌ Command failed: {cmd}")
                if e.stderr:
                    print(f"Error: {e.stderr[:200]}")
            return False
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out: {cmd}")
            return False
    
    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary"""
        
        summary = results['summary']
        
        print("\\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        # Overall status
        status_icons = {
            'excellent': '🎉',
            'good': '✅', 
            'fair': '⚠️ ',
            'poor': '❌'
        }
        
        status = summary['overall_status']
        icon = status_icons.get(status, '❓')
        print(f"Overall Status: {icon} {status.upper()}")
        
        # Critical issues
        if summary['critical_issues']:
            print(f"\\n❌ Critical Issues ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues']:
                print(f"  - {issue}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Next steps based on status
        print(f"\\n🎯 Next Steps:")
        if status == 'excellent':
            print("  Environment is ready! You can start training:")
            print("  python train.py")
        elif status == 'good':
            print("  Environment is mostly ready. Address warnings if needed:")
            print("  python train.py")
        else:
            print("  Fix critical issues using this script's options")

def main():
    """Main setup interface"""
    
    setup = AwareNetMasterSetup()
    
    print("=== AWARE-NET Master Setup ===")
    print("Complete environment setup and dataset preparation")
    print()
    
    while True:
        print("Available Options:")
        print("1. 🚀 Full Setup (PyTorch + Dependencies + Dataset)")
        print("2. 🔥 Install/Update PyTorch (GPU optimized)")
        print("3. 📦 Install Dependencies")
        print("4. 📊 Prepare Dataset")
        print("5. 🔍 Validate Environment")
        print("6. ❓ Show GPU Info")
        print("7. 🏃 Quick Start Training")
        print("8. 🔧 Fix PyTorch Issues (Clean Reinstall)")
        print("9. 📄 Generate Setup Report")
        print("10. 🚪 Exit")
        
        choice = input("\\nSelect option (1-10): ").strip()
        
        if choice == "1":
            print("🚀 Starting full setup...")
            
            # Step 1: PyTorch
            print("\\nStep 1/3: Installing PyTorch...")
            pytorch_success = setup.install_pytorch_optimized()
            
            # Step 2: Dependencies
            print("\\nStep 2/3: Installing dependencies...")
            deps_success = setup.install_dependencies()
            
            # Step 3: Dataset
            print("\\nStep 3/3: Preparing dataset...")
            dataset_success = setup.prepare_dataset()
            
            # Final validation
            print("\\nStep 4/4: Final validation...")
            results = setup.validate_environment()
            setup.print_summary(results)
            
            if results['summary']['overall_status'] in ['excellent', 'good']:
                print("\\n🎉 Full setup completed successfully!")
                print("You can now run: python train.py")
            else:
                print("\\n⚠️  Setup completed with issues. Check recommendations above.")
            
        elif choice == "2":
            gpu_info = setup.detect_gpu_architecture()
            if gpu_info:
                print(f"🎮 Detected: {gpu_info['name']}")
                print(f"📊 Strategy: {gpu_info['pytorch_strategy']}")
            
            setup.install_pytorch_optimized()
            
        elif choice == "3":
            setup.install_dependencies()
            
        elif choice == "4":
            setup.prepare_dataset()
            
        elif choice == "5":
            results = setup.validate_environment()
            setup.print_summary(results)
            
        elif choice == "6":
            gpu_info = setup.detect_gpu_architecture()
            if gpu_info:
                print(f"🎮 GPU: {gpu_info['name']}")
                print(f"📊 Compute Capability: {gpu_info['capability']}")
                print(f"💾 Memory: {gpu_info['memory_gb']:.1f} GB")
                print(f"🚀 Recommended Strategy: {gpu_info['pytorch_strategy']}")
            else:
                print("🎮 No compatible GPU detected")
                
        elif choice == "7":
            print("🏃 Quick start training...")
            print("Checking environment first...")
            
            results = setup.validate_environment()
            if results['summary']['overall_status'] in ['excellent', 'good']:
                print("✅ Environment ready!")
                print("Launching training interface...")
                
                # Launch train.py
                try:
                    import subprocess
                    subprocess.run([sys.executable, "train.py"], check=True)
                except subprocess.CalledProcessError:
                    print("❌ Training script failed")
                except FileNotFoundError:
                    print("❌ train.py not found")
                except KeyboardInterrupt:
                    print("⏹️ Training interrupted")
            else:
                print("❌ Environment not ready for training")
                print("Please run full setup (option 1) first")
                
        elif choice == "8":
            print("🔧 Fixing PyTorch installation issues...")
            print("This will perform a clean reinstall with comprehensive cleanup")
            
            confirm = input("Continue with PyTorch fix? (y/N): ").strip().lower()
            if confirm == 'y':
                setup.fix_pytorch_installation()
            else:
                print("Skipped PyTorch fix")
            
        elif choice == "9":
            print("📄 Generating comprehensive setup report...")
            results = setup.validate_environment()
            
            # Save report
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_path = setup.project_root / "logs" / f"setup_report_{timestamp}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Report saved: {report_path}")
            setup.print_summary(results)
            
        elif choice == "10":
            print("👋 Setup complete!")
            break
            
        else:
            print("❌ Invalid choice, please try again")
        
        print("\\n" + "="*60 + "\\n")

if __name__ == "__main__":
    main()