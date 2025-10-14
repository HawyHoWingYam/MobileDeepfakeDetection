#!/usr/bin/env python3
"""
AWARE-NET Training Script
Simplified training interface with built-in environment checking
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class AwareNetTrainer:
    """Simplified training interface for AWARE-NET"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        
        # Add src to path for imports
        sys.path.append(str(self.project_root / "src"))
    
    def check_environment_quick(self) -> bool:
        """Quick environment check before training"""
        
        print("🔍 Quick Environment Check...")
        
        # Check PyTorch
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            # Check GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                arch = f"sm_{capability[0]}{capability[1]}"
                
                try:
                    # Test GPU functionality
                    test_tensor = torch.zeros(1).cuda()
                    print(f"✅ GPU: {gpu_name} ({arch}) - Functional")
                    gpu_working = True
                except Exception as e:
                    print(f"⚠️  GPU: {gpu_name} ({arch}) - Error: {str(e)[:50]}")
                    print("    Will use CPU training")
                    gpu_working = False
            else:
                print("⚠️  No CUDA GPU detected - will use CPU")
                gpu_working = False
                
        except ImportError:
            print("❌ PyTorch not installed")
            return False
        
        # Check core dependencies
        required_deps = ['torchmetrics', 'timm', 'albumentations', 'pandas', 'numpy']
        missing_deps = []
        
        for dep in required_deps:
            try:
                __import__(dep)
                print(f"✅ {dep}")
            except ImportError:
                missing_deps.append(dep)
                print(f"❌ {dep}")
        
        if missing_deps:
            print(f"\\n❌ Missing dependencies: {', '.join(missing_deps)}")
            print("Run: python setup.py (option 3) to install")
            return False
        
        # Check manifests
        manifests_dir = self.project_root / "manifests"
        if manifests_dir.exists() and list(manifests_dir.glob("*.csv")):
            print("✅ Dataset manifests found")
        else:
            print("⚠️  No dataset manifests found")
            print("Run: python setup.py (option 4) to prepare dataset")
        
        print("✅ Environment check passed")
        return True
    
    def show_training_options(self):
        """Show available training configurations"""
        
        print("🎯 Training Options:")
        print()
        print("1. 🚀 Quick Test (3 epochs, batch=4)")
        print("   Duration: ~15 minutes")
        print("   Purpose: Test environment and training pipeline")
        print()
        print("2. 📈 Small Training (10 epochs, batch=8)")
        print("   Duration: ~1 hour")
        print("   Purpose: Basic model validation")
        print()
        print("3. 🎯 Standard Training (20 epochs, batch=16)")
        print("   Duration: ~3 hours")
        print("   Purpose: Good baseline performance")
        print()
        print("4. 🏆 Full Training (50 epochs, batch=32)")
        print("   Duration: ~8 hours")
        print("   Purpose: Best performance")
        print()
        print("5. ⚙️  Custom Training")
        print("   Duration: Variable")
        print("   Purpose: Custom parameters")
        print()
        print("6. 📊 Resume Training")
        print("   Duration: Variable")
        print("   Purpose: Continue from checkpoint")
        
    def get_training_config(self, choice: str) -> dict:
        """Get training configuration based on user choice"""
        
        configs = {
            "1": {
                "epochs": 3,
                "batch_size": 4,
                "experiment_name": "quick_test",
                "description": "Quick pipeline test"
            },
            "2": {
                "epochs": 10,
                "batch_size": 8,
                "experiment_name": "small_training",
                "description": "Small training run"
            },
            "3": {
                "epochs": 20,
                "batch_size": 16,
                "experiment_name": "standard_training",
                "description": "Standard training"
            },
            "4": {
                "epochs": 50,
                "batch_size": 32,
                "experiment_name": "full_training",
                "description": "Full training"
            }
        }
        
        if choice == "5":
            # Custom configuration
            print("⚙️  Custom Training Configuration:")
            try:
                epochs = int(input("Number of epochs (default 20): ") or "20")
                batch_size = int(input("Batch size (default 16): ") or "16")
                exp_name = input("Experiment name (default custom): ") or "custom"
                
                return {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "experiment_name": exp_name,
                    "description": "Custom training configuration"
                }
            except ValueError:
                print("❌ Invalid input, using default configuration")
                return configs["3"]  # Default to standard
        
        elif choice == "6":
            # Resume training
            experiments_dir = self.project_root / "experiments"
            if not experiments_dir.exists():
                print("❌ No experiments directory found")
                return None
            
            # List available experiments
            exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
            if not exp_dirs:
                print("❌ No previous experiments found")
                return None
            
            print("📂 Available experiments:")
            for i, exp_dir in enumerate(exp_dirs, 1):
                print(f"  {i}. {exp_dir.name}")
            
            try:
                choice_idx = int(input("Select experiment to resume (number): ")) - 1
                if 0 <= choice_idx < len(exp_dirs):
                    selected_exp = exp_dirs[choice_idx]
                    
                    # Look for checkpoint
                    checkpoint_dir = selected_exp / "checkpoints"
                    if checkpoint_dir.exists():
                        checkpoints = list(checkpoint_dir.glob("*.pth"))
                        if checkpoints:
                            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                            
                            return {
                                "resume_path": str(latest_checkpoint),
                                "experiment_name": f"{selected_exp.name}_resumed",
                                "description": f"Resumed from {selected_exp.name}"
                            }
                    
                    print("❌ No checkpoints found in selected experiment")
                    return None
                else:
                    print("❌ Invalid selection")
                    return None
                    
            except (ValueError, IndexError):
                print("❌ Invalid input")
                return None
        
        return configs.get(choice)
    
    def start_training(self, config: dict) -> bool:
        """Start training with given configuration"""
        
        print(f"🚀 Starting Training: {config['description']}")
        print("=" * 60)
        
        # Build command
        cmd_parts = [
            sys.executable,
            "src/stage_00/train_baseline.py"
        ]
        
        # Add training parameters
        if "epochs" in config:
            cmd_parts.extend(["--epochs", str(config["epochs"])])
        
        if "batch_size" in config:
            cmd_parts.extend(["--batch-size", str(config["batch_size"])])
        
        if "experiment_name" in config:
            cmd_parts.extend(["--experiment-name", config["experiment_name"]])
        
        if "resume_path" in config:
            cmd_parts.extend(["--resume", config["resume_path"]])
        
        cmd = " ".join(cmd_parts)
        
        print(f"Command: {cmd}")
        print(f"Working Directory: {self.project_root}")
        
        if "epochs" in config:
            print(f"Epochs: {config['epochs']}")
            print(f"Batch Size: {config['batch_size']}")
        
        print("=" * 60)
        print()
        
        # Change to project directory
        os.chdir(self.project_root)
        
        # Start training
        try:
            # Use subprocess to run training
            result = subprocess.run(cmd, shell=True)
            
            if result.returncode == 0:
                print("\\n✅ Training completed successfully!")
                
                # Show experiment directory
                exp_name = config.get("experiment_name", "unknown")
                experiments_dir = self.project_root / "experiments"
                
                if experiments_dir.exists():
                    # Find the latest experiment directory
                    exp_dirs = [d for d in experiments_dir.iterdir() 
                              if d.is_dir() and exp_name in d.name]
                    
                    if exp_dirs:
                        latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
                        print(f"📁 Results saved to: {latest_exp}")
                        
                        # Show quick results if available
                        results_file = latest_exp / "results" / "experiment_result.json"
                        if results_file.exists():
                            try:
                                import json
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                
                                if 'test_metrics' in results:
                                    test_metrics = results['test_metrics']
                                    print(f"📊 Final Test Results:")
                                    print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                                    print(f"   AUC: {test_metrics.get('auc', 0):.4f}")
                                    print(f"   F1: {test_metrics.get('f1', 0):.4f}")
                            except:
                                pass
                
                return True
            else:
                print(f"\\n❌ Training failed (exit code: {result.returncode})")
                return False
                
        except KeyboardInterrupt:
            print("\\n⏹️ Training interrupted by user")
            return False
        except Exception as e:
            print(f"\\n❌ Training error: {e}")
            return False
    
    def show_experiment_results(self):
        """Show results from previous experiments"""
        
        experiments_dir = self.project_root / "experiments"
        if not experiments_dir.exists():
            print("❌ No experiments directory found")
            return
        
        exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        if not exp_dirs:
            print("❌ No experiments found")
            return
        
        print("📊 Previous Experiments:")
        print("=" * 50)
        
        for exp_dir in sorted(exp_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
            print(f"\\n📁 {exp_dir.name}")
            
            # Check for results
            results_file = exp_dir / "results" / "experiment_result.json"
            if results_file.exists():
                try:
                    import json
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    if 'test_metrics' in results:
                        metrics = results['test_metrics']
                        print(f"   Test Accuracy: {metrics.get('accuracy', 0):.4f}")
                        print(f"   Test AUC: {metrics.get('auc', 0):.4f}")
                        print(f"   Test F1: {metrics.get('f1', 0):.4f}")
                    
                    if 'training_duration' in results:
                        duration = results['training_duration']
                        print(f"   Duration: {duration}")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not read results: {e}")
            else:
                print("   📝 No results file found")

def main():
    """Main training interface"""
    
    trainer = AwareNetTrainer()
    
    print("=== AWARE-NET Training Interface ===")
    print("Simplified training with built-in environment checking")
    print()
    
    while True:
        print("Available Actions:")
        print("1. 🚀 Start Training")
        print("2. 🔍 Check Environment")
        print("3. 📊 Show Previous Results")
        print("4. 🛠️  Run Setup (if needed)")
        print("5. 🚪 Exit")
        
        choice = input("\\nSelect action (1-5): ").strip()
        
        if choice == "1":
            # Quick environment check first
            if not trainer.check_environment_quick():
                print("\\n❌ Environment check failed!")
                print("💡 Try option 4 to run setup, or python setup.py")
                continue
            
            print()
            trainer.show_training_options()
            
            train_choice = input("\\nSelect training option (1-6): ").strip()
            config = trainer.get_training_config(train_choice)
            
            if config:
                success = trainer.start_training(config)
                if success:
                    print("\\n🎉 Training session completed!")
                    
                    # Ask if user wants to see results
                    show_results = input("\\nShow experiment results? (y/N): ").strip().lower()
                    if show_results == 'y':
                        trainer.show_experiment_results()
            else:
                print("❌ Invalid configuration")
        
        elif choice == "2":
            trainer.check_environment_quick()
        
        elif choice == "3":
            trainer.show_experiment_results()
        
        elif choice == "4":
            print("🛠️  Running setup script...")
            try:
                subprocess.run([sys.executable, "setup.py"], check=True)
            except subprocess.CalledProcessError:
                print("❌ Setup script failed")
            except FileNotFoundError:
                print("❌ setup.py not found in current directory")
        
        elif choice == "5":
            print("👋 Happy training!")
            break
        
        else:
            print("❌ Invalid choice, please try again")
        
        print("\\n" + "="*60 + "\\n")

if __name__ == "__main__":
    main()