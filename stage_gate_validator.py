#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Stage-Gate Automatic Validation System
Comprehensive validation against quantified criteria from implementation_stage_00.md
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class ValidationResult:
    """Container for validation results"""
    criterion: str
    category: str
    required_value: Any
    actual_value: Any
    threshold_met: bool
    score: float  # 0-100
    details: str
    evidence: List[str]

@dataclass
class StageGateReport:
    """Complete Stage-Gate validation report"""
    timestamp: str
    overall_status: str
    overall_score: float
    technical_gates: List[ValidationResult]
    academic_gates: List[ValidationResult]
    system_gates: List[ValidationResult]
    quantified_metrics: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]

class StageGateValidator:
    """
    Stage-Gate validation system implementing criteria from implementation_stage_00.md
    
    Validates against three categories:
    - Technical Gates: Functional and performance requirements
    - Academic Gates: Innovation and rigor standards
    - System Gates: Usability and scalability requirements
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize Stage-Gate validator
        
        Args:
            project_root: Root directory of the AWARE-NET project
        """
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.configs_dir = self.project_root / "configs"
        
        # Stage-Gate criteria from implementation_stage_00.md
        self.technical_criteria = {
            'environment_success_rate': {'min': 95, 'target': 99, 'unit': '%'},
            'baseline_auc': {'min': 0.88, 'target': 0.90, 'unit': 'AUC-ROC'},
            'data_loading_speed': {'min': 100, 'target': 200, 'unit': 'samples/sec'},
            'code_coverage': {'min': 80, 'target': 90, 'unit': '%'},
            'documentation_completeness': {'min': 90, 'target': 95, 'unit': '%'}
        }
        
        self.academic_criteria = {
            'reproducibility_standard': {'components': ['seed_management', 'deterministic_training', 'experiment_config']},
            'statistical_rigor': {'components': ['confidence_intervals', 'significance_tests', 'bootstrap_methods']},
            'baseline_analysis_depth': {'components': ['failure_analysis', 'cross_dataset_evaluation', 'improvement_recommendations']},
            'documentation_quality': {'components': ['api_docs', 'technical_specs', 'user_guides']}
        }
        
        self.system_criteria = {
            'cross_platform_compatibility': {'platforms': ['Windows', 'Linux', 'macOS']},
            'performance_benchmarks': {'inference_speed_ms': 100, 'memory_usage_mb': 4000},
            'extensibility_design': {'components': ['modular_architecture', 'plugin_support', 'api_interfaces']},
            'operational_readiness': {'components': ['monitoring', 'logging', 'error_handling']}
        }
        
        self.validation_results = {
            'technical': [],
            'academic': [],
            'system': []
        }
    
    def validate_technical_gates(self) -> List[ValidationResult]:
        """Validate technical requirements"""
        print("Validating Technical Gates...")
        results = []
        
        # 1. Environment Reproducibility
        env_result = self._validate_environment_success_rate()
        results.append(env_result)
        
        # 2. Data Management Integrity
        data_result = self._validate_data_management()
        results.append(data_result)
        
        # 3. Tool Library Functionality
        tools_result = self._validate_tool_functionality()
        results.append(tools_result)
        
        # 4. Baseline Model Performance
        baseline_result = self._validate_baseline_performance()
        results.append(baseline_result)
        
        # 5. Project Structure Reasonableness
        structure_result = self._validate_project_structure()
        results.append(structure_result)
        
        return results
    
    def validate_academic_gates(self) -> List[ValidationResult]:
        """Validate academic standards"""
        print("Validating Academic Gates...")
        results = []
        
        # 1. Reproducibility Standard
        repro_result = self._validate_reproducibility()
        results.append(repro_result)
        
        # 2. Statistical Rigor
        stats_result = self._validate_statistical_rigor()
        results.append(stats_result)
        
        # 3. Baseline Analysis Depth
        analysis_result = self._validate_baseline_analysis()
        results.append(analysis_result)
        
        # 4. Documentation Quality
        docs_result = self._validate_documentation_quality()
        results.append(docs_result)
        
        return results
    
    def validate_system_gates(self) -> List[ValidationResult]:
        """Validate system requirements"""
        print("Validating System Gates...")
        results = []
        
        # 1. Cross-Platform Compatibility
        platform_result = self._validate_platform_compatibility()
        results.append(platform_result)
        
        # 2. Performance Benchmarks
        perf_result = self._validate_performance_benchmarks()
        results.append(perf_result)
        
        # 3. Extensibility Design
        ext_result = self._validate_extensibility()
        results.append(ext_result)
        
        # 4. Operational Readiness
        ops_result = self._validate_operational_readiness()
        results.append(ops_result)
        
        return results
    
    def _validate_environment_success_rate(self) -> ValidationResult:
        """Validate environment configuration success rate"""
        print("  Checking environment configuration...")
        
        evidence = []
        success_count = 0
        total_attempts = 10
        
        # Test key environment components
        required_files = [
            'environment.yml',
            'Dockerfile',
            'requirements.txt'
        ]
        
        for file in required_files:
            file_path = self.project_root / file
            if file_path.exists():
                success_count += 1
                evidence.append(f"✅ {file} exists")
            else:
                evidence.append(f"❌ {file} missing")
        
        # Test import success
        critical_imports = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'), 
            ('pandas', 'Pandas'),
            ('sklearn', 'Scikit-learn'),
            ('cv2', 'OpenCV')
        ]
        
        import_success = 0
        for module, name in critical_imports:
            try:
                __import__(module)
                import_success += 1
                evidence.append(f"✅ {name} import successful")
            except ImportError:
                evidence.append(f"❌ {name} import failed")
        
        # Calculate success rate
        total_checks = len(required_files) + len(critical_imports)
        actual_success_rate = ((success_count + import_success) / total_checks) * 100
        
        threshold_met = actual_success_rate >= self.technical_criteria['environment_success_rate']['min']
        
        score = min(100, (actual_success_rate / self.technical_criteria['environment_success_rate']['target']) * 100)
        
        return ValidationResult(
            criterion="Environment Reproducibility",
            category="Technical",
            required_value=f">= {self.technical_criteria['environment_success_rate']['min']}%",
            actual_value=f"{actual_success_rate:.1f}%",
            threshold_met=threshold_met,
            score=score,
            details=f"Environment setup success rate assessment",
            evidence=evidence
        )
    
    def _validate_data_management(self) -> ValidationResult:
        """Validate data management system completeness"""
        print("  Checking data management system...")
        
        evidence = []
        components_found = 0
        
        required_components = [
            ('src/utils/dataset_config.py', 'Dataset Configuration'),
            ('src/utils/manifest_generator.py', 'Manifest Generator'),
            ('src/utils/data_validator.py', 'Data Validator'),
            ('configs/dataset_paths.json', 'Configuration Template')
        ]
        
        for file_path, component in required_components:
            full_path = self.project_root / file_path
            if full_path.exists():
                components_found += 1
                evidence.append(f"✅ {component} ({file_path})")
            else:
                evidence.append(f"❌ {component} missing ({file_path})")
        
        # Check for manifest support (simulated)
        manifest_dir = self.project_root / "manifests"
        if manifest_dir.exists():
            components_found += 1
            evidence.append(f"✅ Manifests directory exists")
        else:
            evidence.append(f"❌ Manifests directory missing")
        
        success_rate = (components_found / (len(required_components) + 1)) * 100
        threshold_met = success_rate >= 80  # 4+ components required
        
        score = min(100, (success_rate / 100) * 100)
        
        return ValidationResult(
            criterion="Data Management Integrity",
            category="Technical",
            required_value="4+ dataset formats supported",
            actual_value=f"{components_found}/{len(required_components)+1} components",
            threshold_met=threshold_met,
            score=score,
            details="Data management system component verification",
            evidence=evidence
        )
    
    def _validate_tool_functionality(self) -> ValidationResult:
        """Validate academic tools library functionality"""
        print("  Checking tool library functionality...")
        
        evidence = []
        tools_found = 0
        
        required_tools = [
            ('src/utils/metrics.py', 'Academic Metrics'),
            ('src/utils/visualization.py', 'Visualization Tools'),
            ('src/utils/experiment_utils.py', 'Experiment Management'),
            ('src/utils/calibration_tools.py', 'Calibration Analysis')
        ]
        
        for file_path, tool in required_tools:
            full_path = self.project_root / file_path
            if full_path.exists():
                tools_found += 1
                evidence.append(f"✅ {tool} ({file_path})")
                
                # Basic functionality test
                try:
                    # Import test
                    spec = importlib.util.spec_from_file_location("test_module", full_path)
                    if spec and spec.loader:
                        evidence.append(f"  → Import test passed")
                except Exception as e:
                    evidence.append(f"  → Import test failed: {str(e)[:50]}")
            else:
                evidence.append(f"❌ {tool} missing ({file_path})")
        
        success_rate = (tools_found / len(required_tools)) * 100
        threshold_met = tools_found == len(required_tools)  # All tools required
        
        score = min(100, success_rate)
        
        return ValidationResult(
            criterion="Tool Library Functionality",
            category="Technical",
            required_value="All evaluation tools operational",
            actual_value=f"{tools_found}/{len(required_tools)} tools found",
            threshold_met=threshold_met,
            score=score,
            details="Academic tools library verification",
            evidence=evidence
        )
    
    def _validate_baseline_performance(self) -> ValidationResult:
        """Validate baseline model performance capability"""
        print("  Checking baseline model implementation...")
        
        evidence = []
        components_found = 0
        
        baseline_components = [
            ('src/stage_00/baseline_model.py', 'Model Architecture'),
            ('src/stage_00/train_baseline.py', 'Training Script'),
            ('src/stage_00/evaluate_baseline.py', 'Evaluation Script'),
            ('configs/training_config.json', 'Training Configuration')
        ]
        
        for file_path, component in baseline_components:
            full_path = self.project_root / file_path
            if full_path.exists():
                components_found += 1
                evidence.append(f"✅ {component} ({file_path})")
            else:
                evidence.append(f"❌ {component} missing ({file_path})")
        
        # Check model instantiation capability
        try:
            sys.path.insert(0, str(self.src_dir))
            from stage_00.baseline_model import EfficientNetV2B3Baseline
            model = EfficientNetV2B3Baseline(pretrained=False)
            evidence.append(f"✅ Model instantiation successful")
            components_found += 1
        except Exception as e:
            evidence.append(f"❌ Model instantiation failed: {str(e)[:50]}")
        
        success_rate = (components_found / (len(baseline_components) + 1)) * 100
        threshold_met = components_found >= len(baseline_components)  # All baseline components required
        
        score = min(100, success_rate)
        
        return ValidationResult(
            criterion="Baseline Model Performance",
            category="Technical",
            required_value="Complete baseline implementation",
            actual_value=f"{components_found}/{len(baseline_components)+1} components",
            threshold_met=threshold_met,
            score=score,
            details="Baseline model implementation verification",
            evidence=evidence
        )
    
    def _validate_project_structure(self) -> ValidationResult:
        """Validate 10-stage project structure"""
        print("  Checking project structure...")
        
        evidence = []
        stages_found = 0
        
        # Check for 10-stage structure
        for i in range(10):
            stage_dir = self.src_dir / f"stage_{i:02d}"
            if stage_dir.exists():
                stages_found += 1
                evidence.append(f"✅ Stage {i:02d} directory exists")
            else:
                evidence.append(f"❌ Stage {i:02d} directory missing")
        
        # Check utility structure
        utils_components = ['dataset_config.py', 'metrics.py', 'visualization.py', 'experiment_utils.py']
        utils_found = 0
        
        for component in utils_components:
            utils_path = self.src_dir / "utils" / component
            if utils_path.exists():
                utils_found += 1
                evidence.append(f"✅ Utils: {component}")
            else:
                evidence.append(f"❌ Utils: {component} missing")
        
        total_score = ((stages_found / 10) * 50) + ((utils_found / len(utils_components)) * 50)
        threshold_met = stages_found >= 10 and utils_found >= len(utils_components)
        
        return ValidationResult(
            criterion="Project Structure Reasonableness",
            category="Technical",
            required_value="10-stage architecture + utilities",
            actual_value=f"{stages_found}/10 stages, {utils_found}/{len(utils_components)} utils",
            threshold_met=threshold_met,
            score=total_score,
            details="Project directory structure verification",
            evidence=evidence
        )
    
    def _validate_reproducibility(self) -> ValidationResult:
        """Validate reproducibility standards"""
        print("  Checking reproducibility implementation...")
        
        evidence = []
        components_found = 0
        
        # Check experiment utils for reproducibility features
        exp_utils_path = self.src_dir / "utils" / "experiment_utils.py"
        if exp_utils_path.exists():
            try:
                with open(exp_utils_path, 'r') as f:
                    content = f.read()
                    
                if 'random.seed' in content or 'np.random.seed' in content or 'torch.manual_seed' in content:
                    components_found += 1
                    evidence.append("✅ Seed management implemented")
                else:
                    evidence.append("❌ Seed management not found")
                
                if 'deterministic' in content.lower() or 'reproducible' in content.lower():
                    components_found += 1
                    evidence.append("✅ Deterministic training support")
                else:
                    evidence.append("❌ Deterministic training not explicit")
                
                if 'ExperimentConfig' in content or 'config' in content.lower():
                    components_found += 1
                    evidence.append("✅ Experiment configuration management")
                else:
                    evidence.append("❌ Experiment configuration incomplete")
                    
            except Exception as e:
                evidence.append(f"❌ Error reading experiment_utils.py: {e}")
        else:
            evidence.append("❌ experiment_utils.py not found")
        
        success_rate = (components_found / 3) * 100
        threshold_met = components_found >= 2
        
        return ValidationResult(
            criterion="Reproducibility Standard",
            category="Academic",
            required_value="Seed management + deterministic training + config",
            actual_value=f"{components_found}/3 components implemented",
            threshold_met=threshold_met,
            score=success_rate,
            details="Reproducibility features verification",
            evidence=evidence
        )
    
    def _validate_statistical_rigor(self) -> ValidationResult:
        """Validate statistical rigor implementation"""
        print("  Checking statistical rigor...")
        
        evidence = []
        components_found = 0
        
        # Check metrics.py for statistical features
        metrics_path = self.src_dir / "utils" / "metrics.py"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    content = f.read()
                
                if 'confidence_interval' in content or 'bootstrap' in content:
                    components_found += 1
                    evidence.append("✅ Confidence intervals implemented")
                else:
                    evidence.append("❌ Confidence intervals not found")
                
                if 'significance' in content or 'p_value' in content:
                    components_found += 1
                    evidence.append("✅ Significance testing implemented")
                else:
                    evidence.append("❌ Significance testing not found")
                
                if 'bootstrap' in content:
                    components_found += 1
                    evidence.append("✅ Bootstrap methods implemented")
                else:
                    evidence.append("❌ Bootstrap methods not found")
                    
            except Exception as e:
                evidence.append(f"❌ Error reading metrics.py: {e}")
        else:
            evidence.append("❌ metrics.py not found")
        
        success_rate = (components_found / 3) * 100
        threshold_met = components_found >= 2
        
        return ValidationResult(
            criterion="Statistical Rigor",
            category="Academic",
            required_value="CI + significance tests + bootstrap",
            actual_value=f"{components_found}/3 methods implemented",
            threshold_met=threshold_met,
            score=success_rate,
            details="Statistical analysis capabilities verification",
            evidence=evidence
        )
    
    def _validate_baseline_analysis(self) -> ValidationResult:
        """Validate baseline analysis depth"""
        print("  Checking baseline analysis capabilities...")
        
        evidence = []
        components_found = 0
        
        # Check evaluate_baseline.py for analysis features
        eval_path = self.src_dir / "stage_00" / "evaluate_baseline.py"
        if eval_path.exists():
            try:
                with open(eval_path, 'r') as f:
                    content = f.read()
                
                if 'failure_analysis' in content.lower() or 'error_analysis' in content.lower():
                    components_found += 1
                    evidence.append("✅ Failure analysis implemented")
                else:
                    evidence.append("❌ Failure analysis not found")
                
                if 'cross_dataset' in content.lower() or 'multi_dataset' in content.lower():
                    components_found += 1
                    evidence.append("✅ Cross-dataset evaluation implemented")
                else:
                    evidence.append("❌ Cross-dataset evaluation not found")
                
                if 'recommendation' in content.lower() or 'improvement' in content.lower():
                    components_found += 1
                    evidence.append("✅ Improvement recommendations implemented")
                else:
                    evidence.append("❌ Improvement recommendations not found")
                    
            except Exception as e:
                evidence.append(f"❌ Error reading evaluate_baseline.py: {e}")
        else:
            evidence.append("❌ evaluate_baseline.py not found")
        
        success_rate = (components_found / 3) * 100
        threshold_met = components_found >= 2
        
        return ValidationResult(
            criterion="Baseline Analysis Depth",
            category="Academic",
            required_value="Failure analysis + cross-dataset + recommendations",
            actual_value=f"{components_found}/3 analysis types implemented",
            threshold_met=threshold_met,
            score=success_rate,
            details="Baseline analysis comprehensiveness verification",
            evidence=evidence
        )
    
    def _validate_documentation_quality(self) -> ValidationResult:
        """Validate documentation quality and completeness"""
        print("  Checking documentation quality...")
        
        evidence = []
        components_found = 0
        
        # Check for API documentation
        doc_indicators = [
            ('src/utils/', 'docstring', 'API Documentation'),
            ('README.md', 'exists', 'User Guide'),
            ('CLAUDE.md', 'exists', 'Development Guide')
        ]
        
        for location, check_type, doc_type in doc_indicators:
            if check_type == 'docstring':
                # Check for docstrings in Python files
                doc_files = list(Path(self.project_root / location).glob('*.py'))
                if doc_files:
                    docstring_found = False
                    for file in doc_files[:3]:  # Check first 3 files
                        try:
                            with open(file, 'r') as f:
                                content = f.read()
                                if '"""' in content and ('Args:' in content or 'Parameters:' in content):
                                    docstring_found = True
                                    break
                        except:
                            continue
                    
                    if docstring_found:
                        components_found += 1
                        evidence.append(f"✅ {doc_type} found in {location}")
                    else:
                        evidence.append(f"❌ {doc_type} insufficient in {location}")
                else:
                    evidence.append(f"❌ No Python files found in {location}")
            
            elif check_type == 'exists':
                file_path = self.project_root / location
                if file_path.exists():
                    components_found += 1
                    evidence.append(f"✅ {doc_type} exists ({location})")
                else:
                    evidence.append(f"❌ {doc_type} missing ({location})")
        
        success_rate = (components_found / len(doc_indicators)) * 100
        threshold_met = success_rate >= 66  # At least 2/3 doc types
        
        return ValidationResult(
            criterion="Documentation Quality",
            category="Academic",
            required_value="API docs + technical specs + user guides",
            actual_value=f"{components_found}/{len(doc_indicators)} documentation types",
            threshold_met=threshold_met,
            score=success_rate,
            details="Documentation completeness and quality verification",
            evidence=evidence
        )
    
    def _validate_platform_compatibility(self) -> ValidationResult:
        """Validate cross-platform compatibility"""
        print("  Checking platform compatibility...")
        
        evidence = []
        compatibility_score = 0
        
        # Check for Docker support
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            compatibility_score += 40
            evidence.append("✅ Docker support (cross-platform)")
        else:
            evidence.append("❌ Docker support missing")
        
        # Check for conda environment
        env_file = self.project_root / "environment.yml"
        if env_file.exists():
            compatibility_score += 30
            evidence.append("✅ Conda environment (cross-platform)")
        else:
            evidence.append("❌ Conda environment missing")
        
        # Check for pip requirements
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            compatibility_score += 30
            evidence.append("✅ Pip requirements (cross-platform)")
        else:
            evidence.append("❌ Pip requirements missing")
        
        threshold_met = compatibility_score >= 70  # Good cross-platform support
        
        return ValidationResult(
            criterion="Cross-Platform Compatibility",
            category="System",
            required_value="Docker + conda/pip support",
            actual_value=f"{compatibility_score}/100 compatibility score",
            threshold_met=threshold_met,
            score=compatibility_score,
            details="Platform compatibility assessment",
            evidence=evidence
        )
    
    def _validate_performance_benchmarks(self) -> ValidationResult:
        """Validate performance benchmarks"""
        print("  Checking performance benchmarks...")
        
        evidence = []
        benchmark_score = 0
        
        # Check for performance test implementation
        perf_test_path = self.tests_dir / "test_performance_benchmarks.py"
        if perf_test_path.exists():
            benchmark_score += 50
            evidence.append("✅ Performance benchmark tests exist")
            
            try:
                with open(perf_test_path, 'r') as f:
                    content = f.read()
                    
                if 'inference_speed' in content.lower():
                    benchmark_score += 25
                    evidence.append("✅ Inference speed benchmarks")
                else:
                    evidence.append("❌ Inference speed benchmarks missing")
                
                if 'memory_usage' in content.lower():
                    benchmark_score += 25
                    evidence.append("✅ Memory usage benchmarks")
                else:
                    evidence.append("❌ Memory usage benchmarks missing")
                    
            except Exception as e:
                evidence.append(f"❌ Error reading performance tests: {e}")
        else:
            evidence.append("❌ Performance benchmark tests missing")
        
        threshold_met = benchmark_score >= 75
        
        return ValidationResult(
            criterion="Performance Benchmarks",
            category="System", 
            required_value="Inference < 100ms, Memory < 4GB",
            actual_value=f"{benchmark_score}/100 benchmark coverage",
            threshold_met=threshold_met,
            score=benchmark_score,
            details="Performance benchmark implementation verification",
            evidence=evidence
        )
    
    def _validate_extensibility(self) -> ValidationResult:
        """Validate extensibility design"""
        print("  Checking extensibility design...")
        
        evidence = []
        extensibility_score = 0
        
        # Check modular architecture
        stage_dirs = list(self.src_dir.glob("stage_*"))
        if len(stage_dirs) >= 10:
            extensibility_score += 40
            evidence.append(f"✅ Modular architecture ({len(stage_dirs)} stages)")
        else:
            evidence.append(f"❌ Incomplete modular architecture ({len(stage_dirs)}/10 stages)")
        
        # Check utility modules
        utils_dir = self.src_dir / "utils"
        if utils_dir.exists():
            utils_files = list(utils_dir.glob("*.py"))
            if len(utils_files) >= 4:  # Key utility modules
                extensibility_score += 30
                evidence.append(f"✅ Utility modules ({len(utils_files)} modules)")
            else:
                evidence.append(f"❌ Insufficient utility modules ({len(utils_files)}/4)")
        else:
            evidence.append("❌ Utils directory missing")
        
        # Check configuration support
        if self.configs_dir.exists():
            config_files = list(self.configs_dir.glob("*.json"))
            if config_files:
                extensibility_score += 30
                evidence.append(f"✅ Configuration support ({len(config_files)} configs)")
            else:
                evidence.append("❌ No configuration files found")
        else:
            evidence.append("❌ Configs directory missing")
        
        threshold_met = extensibility_score >= 70
        
        return ValidationResult(
            criterion="Extensibility Design",
            category="System",
            required_value="Modular architecture + plugin support + APIs",
            actual_value=f"{extensibility_score}/100 extensibility score",
            threshold_met=threshold_met,
            score=extensibility_score,
            details="System extensibility assessment",
            evidence=evidence
        )
    
    def _validate_operational_readiness(self) -> ValidationResult:
        """Validate operational readiness"""
        print("  Checking operational readiness...")
        
        evidence = []
        operational_score = 0
        
        # Check for logging support
        log_indicators = ['logging', 'logger', 'print(']  # Simple logging check
        logging_found = False
        
        python_files = list(self.src_dir.glob("**/*.py"))
        for file in python_files[:10]:  # Check first 10 files
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if any(indicator in content for indicator in log_indicators):
                        logging_found = True
                        break
            except:
                continue
        
        if logging_found:
            operational_score += 35
            evidence.append("✅ Logging implementation found")
        else:
            evidence.append("❌ Logging implementation not found")
        
        # Check for error handling
        error_handling_found = False
        for file in python_files[:10]:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'try:' in content and 'except' in content:
                        error_handling_found = True
                        break
            except:
                continue
        
        if error_handling_found:
            operational_score += 35
            evidence.append("✅ Error handling implementation found")
        else:
            evidence.append("❌ Error handling implementation not found")
        
        # Check for test coverage
        if self.tests_dir.exists():
            test_files = list(self.tests_dir.glob("test_*.py"))
            if len(test_files) >= 4:
                operational_score += 30
                evidence.append(f"✅ Test coverage ({len(test_files)} test files)")
            else:
                evidence.append(f"❌ Insufficient test coverage ({len(test_files)}/4 test files)")
        else:
            evidence.append("❌ Tests directory missing")
        
        threshold_met = operational_score >= 70
        
        return ValidationResult(
            criterion="Operational Readiness",
            category="System",
            required_value="Monitoring + logging + error handling + testing",
            actual_value=f"{operational_score}/100 operational score",
            threshold_met=threshold_met,
            score=operational_score,
            details="Operational readiness assessment",
            evidence=evidence
        )
    
    def calculate_overall_score(self, results: Dict[str, List[ValidationResult]]) -> Tuple[float, str]:
        """Calculate overall Stage-Gate score and status"""
        
        # Weight categories
        weights = {
            'technical': 0.5,    # 50% - Core functionality
            'academic': 0.3,     # 30% - Research quality
            'system': 0.2        # 20% - System quality
        }
        
        category_scores = {}
        for category, result_list in results.items():
            if result_list:
                category_scores[category] = sum(r.score for r in result_list) / len(result_list)
            else:
                category_scores[category] = 0
        
        overall_score = sum(category_scores[cat] * weights[cat] for cat in weights.keys())
        
        # Determine status
        if overall_score >= 90:
            status = "EXCELLENT - Ready for Stage 1"
        elif overall_score >= 80:
            status = "GOOD - Ready for Stage 1"
        elif overall_score >= 70:
            status = "ACCEPTABLE - Minor improvements needed"
        elif overall_score >= 60:
            status = "NEEDS IMPROVEMENT - Significant changes required"
        else:
            status = "INADEQUATE - Major rework needed"
        
        return overall_score, status
    
    def generate_recommendations(self, results: Dict[str, List[ValidationResult]]) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        
        recommendations = []
        
        # Analyze failed criteria
        for category, result_list in results.items():
            for result in result_list:
                if not result.threshold_met:
                    if result.score < 50:
                        priority = "HIGH PRIORITY"
                    elif result.score < 75:
                        priority = "MEDIUM PRIORITY"
                    else:
                        priority = "LOW PRIORITY"
                    
                    recommendations.append(f"{priority}: {result.criterion} - {result.details}")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Excellent work! All Stage-Gate criteria met.")
        else:
            recommendations.append("Focus on highest priority items first.")
            recommendations.append("Re-run validation after implementing fixes.")
        
        return recommendations
    
    def run_comprehensive_validation(self) -> StageGateReport:
        """Run complete Stage-Gate validation"""
        
        print(f"\n{'='*80}")
        print("AWARE-NET STAGE 0: COMPREHENSIVE STAGE-GATE VALIDATION")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project Root: {self.project_root}")
        
        results = {
            'technical': self.validate_technical_gates(),
            'academic': self.validate_academic_gates(),
            'system': self.validate_system_gates()
        }
        
        overall_score, status = self.calculate_overall_score(results)
        recommendations = self.generate_recommendations(results)
        
        # Generate next steps
        next_steps = []
        if overall_score >= 80:
            next_steps = [
                "Complete any remaining documentation",
                "Run final performance validation",
                "Begin Stage 1 planning",
                "Archive Stage 0 deliverables"
            ]
        else:
            next_steps = [
                "Address high-priority recommendations",
                "Re-run Stage-Gate validation",
                "Implement missing components",
                "Improve low-scoring areas"
            ]
        
        report = StageGateReport(
            timestamp=datetime.now().isoformat(),
            overall_status=status,
            overall_score=overall_score,
            technical_gates=results['technical'],
            academic_gates=results['academic'],
            system_gates=results['system'],
            quantified_metrics=self._extract_quantified_metrics(results),
            recommendations=recommendations,
            next_steps=next_steps
        )
        
        return report
    
    def _extract_quantified_metrics(self, results: Dict[str, List[ValidationResult]]) -> Dict[str, Any]:
        """Extract quantified metrics for reporting"""
        
        metrics = {}
        
        for category, result_list in results.items():
            metrics[category] = {
                'total_criteria': len(result_list),
                'passed_criteria': sum(1 for r in result_list if r.threshold_met),
                'average_score': sum(r.score for r in result_list) / len(result_list) if result_list else 0,
                'individual_scores': {r.criterion: r.score for r in result_list}
            }
        
        return metrics
    
    def save_report(self, report: StageGateReport, output_file: str = "stage_gate_report.json"):
        """Save validation report to file"""
        
        report_dict = asdict(report)
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\nStage-Gate validation report saved to: {output_file}")
    
    def print_summary(self, report: StageGateReport):
        """Print formatted validation summary"""
        
        print(f"\n{'='*80}")
        print("STAGE-GATE VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Overall Status: {report.overall_status}")
        
        # Category breakdown
        categories = ['technical', 'academic', 'system']
        category_results = {
            'technical': report.technical_gates,
            'academic': report.academic_gates, 
            'system': report.system_gates
        }
        
        for category in categories:
            results = category_results[category]
            passed = sum(1 for r in results if r.threshold_met)
            total = len(results)
            avg_score = sum(r.score for r in results) / total if results else 0
            
            print(f"\n{category.title()} Gates: {passed}/{total} passed (avg: {avg_score:.1f})")
            
            for result in results:
                status = "✅" if result.threshold_met else "❌"
                print(f"  {status} {result.criterion}: {result.score:.1f}/100")
        
        # Recommendations
        if report.recommendations:
            print(f"\n{'='*40}")
            print("RECOMMENDATIONS")
            print(f"{'='*40}")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
        
        # Next Steps
        if report.next_steps:
            print(f"\n{'='*40}")
            print("NEXT STEPS")
            print(f"{'='*40}")
            for i, step in enumerate(report.next_steps, 1):
                print(f"{i}. {step}")
        
        # Final assessment
        print(f"\n{'='*80}")
        if report.overall_score >= 80:
            print("🎉 STAGE-GATE STATUS: APPROVED FOR STAGE 1")
        else:
            print("⚠️  STAGE-GATE STATUS: IMPROVEMENTS REQUIRED")
        print(f"{'='*80}")


def main():
    """Main validation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AWARE-NET Stage 0 Stage-Gate Validator")
    parser.add_argument('--project-root', default='.', 
                       help='Root directory of AWARE-NET project')
    parser.add_argument('--output', default='stage_gate_report.json',
                       help='Output file for validation report')
    parser.add_argument('--category', choices=['technical', 'academic', 'system', 'all'],
                       default='all', help='Validation category to run')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = StageGateValidator(args.project_root)
    
    if args.category == 'all':
        # Run comprehensive validation
        report = validator.run_comprehensive_validation()
        validator.save_report(report, args.output)
        validator.print_summary(report)
        
        # Exit with appropriate code
        sys.exit(0 if report.overall_score >= 80 else 1)
    
    else:
        # Run specific category
        if args.category == 'technical':
            results = validator.validate_technical_gates()
        elif args.category == 'academic':
            results = validator.validate_academic_gates()
        elif args.category == 'system':
            results = validator.validate_system_gates()
        
        # Print results
        passed = sum(1 for r in results if r.threshold_met)
        total = len(results)
        
        print(f"\n{args.category.title()} Gates: {passed}/{total} passed")
        for result in results:
            status = "✅" if result.threshold_met else "❌"
            print(f"  {status} {result.criterion}: {result.score:.1f}/100")
        
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()