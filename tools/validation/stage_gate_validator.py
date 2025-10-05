#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Stage-Gate Automatic Validation System
Comprehensive validation against quantified criteria from project specifications
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime

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
    Stage-Gate validation system implementing criteria for Stage 0 completion

    Validates against three categories:
    - Technical Gates: Functional and performance requirements
    - Academic Gates: Innovation and rigor standards
    - System Gates: Usability and scalability requirements
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.manifests_dir = self.project_root / "manifests"
        self.configs_dir = self.project_root / "configs"

    def validate_technical_gates(self) -> List[ValidationResult]:
        """Validate technical requirements"""
        print("🔧 Validating Technical Gates...")
        results = []

        # 1. Dataset Management
        results.append(self._validate_dataset_management())

        # 2. Model Architecture
        results.append(self._validate_model_architecture())

        # 3. Training Pipeline
        results.append(self._validate_training_pipeline())

        # 4. Academic Tools
        results.append(self._validate_academic_tools())

        # 5. Environment Setup
        results.append(self._validate_environment_setup())

        return results

    def validate_academic_gates(self) -> List[ValidationResult]:
        """Validate academic standards"""
        print("🎓 Validating Academic Gates...")
        results = []

        # 1. Reproducibility
        results.append(self._validate_reproducibility())

        # 2. Statistical Rigor
        results.append(self._validate_statistical_rigor())

        # 3. Documentation Quality
        results.append(self._validate_documentation())

        # 4. Baseline Analysis Capability
        results.append(self._validate_baseline_analysis())

        return results

    def validate_system_gates(self) -> List[ValidationResult]:
        """Validate system requirements"""
        print("⚙️ Validating System Gates...")
        results = []

        # 1. Project Structure
        results.append(self._validate_project_structure())

        # 2. Configuration Management
        results.append(self._validate_configuration_management())

        # 3. Extensibility
        results.append(self._validate_extensibility())

        # 4. Operational Readiness
        results.append(self._validate_operational_readiness())

        return results

    def _validate_dataset_management(self) -> ValidationResult:
        """Validate dataset management system"""
        evidence = []
        components_found = 0

        # Check manifests
        required_manifests = [
            "celebdf_v2_train.csv", "celebdf_v2_val.csv", "celebdf_v2_test.csv",
            "faceforensics_train.csv", "faceforensics_val.csv", "faceforensics_test.csv",
            "deeperforensics_train.csv", "deeperforensics_val.csv", "deeperforensics_test.csv"
        ]

        manifest_count = 0
        for manifest in required_manifests:
            if (self.manifests_dir / manifest).exists():
                manifest_count += 1

        evidence.append(f"Manifests: {manifest_count}/{len(required_manifests)} found")
        components_found += min(manifest_count / len(required_manifests), 1.0) * 40

        # Check data management utilities
        utils_files = [
            "src/utils/dataset_config.py",
            "src/utils/manifest_generator.py",
            "src/utils/data_validator.py"
        ]

        utils_count = 0
        for util_file in utils_files:
            if (self.project_root / util_file).exists():
                utils_count += 1

        evidence.append(f"Utilities: {utils_count}/{len(utils_files)} found")
        components_found += (utils_count / len(utils_files)) * 40

        # Check configuration
        if (self.configs_dir / "unified_dataset_config.json").exists():
            components_found += 20
            evidence.append("✅ Unified dataset configuration exists")
        else:
            evidence.append("❌ Unified dataset configuration missing")

        threshold_met = components_found >= 80

        return ValidationResult(
            criterion="Dataset Management System",
            category="Technical",
            required_value="Complete dataset pipeline",
            actual_value=f"{components_found:.1f}/100 completeness",
            threshold_met=threshold_met,
            score=components_found,
            details="Dataset manifests, utilities, and configuration",
            evidence=evidence
        )

    def _validate_model_architecture(self) -> ValidationResult:
        """Validate model architecture implementation"""
        evidence = []
        score = 0

        try:
            # Test model import
            sys.path.insert(0, str(self.src_dir))
            from stage_00.baseline_model import EfficientNetV2B3Baseline
            evidence.append("✅ Model class imports successfully")
            score += 30

            # Test B3 support
            try:
                model = EfficientNetV2B3Baseline(
                    num_classes=1,
                    pretrained=False,
                    model_name='tf_efficientnetv2_b3'
                )
                evidence.append("✅ EfficientNetV2-B3 model instantiation works")
                score += 40
            except Exception as e:
                evidence.append(f"❌ B3 instantiation failed: {str(e)[:50]}")

            # Test forward pass
            try:
                import torch
                x = torch.randn(1, 3, 256, 256)
                with torch.no_grad():
                    output = model(x)
                if output.shape == (1, 1):
                    evidence.append("✅ Model forward pass produces correct output shape")
                    score += 30
                else:
                    evidence.append(f"❌ Wrong output shape: {output.shape}")
            except Exception as e:
                evidence.append(f"❌ Forward pass failed: {str(e)[:50]}")

        except ImportError as e:
            evidence.append(f"❌ Model import failed: {str(e)[:50]}")

        threshold_met = score >= 70

        return ValidationResult(
            criterion="Model Architecture",
            category="Technical",
            required_value="EfficientNetV2-B3 working implementation",
            actual_value=f"{score}/100 functionality",
            threshold_met=threshold_met,
            score=score,
            details="Model instantiation and forward pass validation",
            evidence=evidence
        )

    def _validate_training_pipeline(self) -> ValidationResult:
        """Validate training pipeline components"""
        evidence = []
        components = 0

        required_files = [
            "src/stage_00/train_baseline.py",
            "src/stage_00/evaluate_baseline.py",
            "src/stage_00/dataset.py"
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                components += 1
                evidence.append(f"✅ {file_path} exists")
            else:
                evidence.append(f"❌ {file_path} missing")

        # Check for MultiDatasetWrapper
        train_file = self.project_root / "src/stage_00/train_baseline.py"
        if train_file.exists():
            try:
                with open(train_file, 'r') as f:
                    content = f.read()
                if 'MultiDatasetWrapper' in content:
                    components += 1
                    evidence.append("✅ MultiDatasetWrapper implementation found")
                else:
                    evidence.append("❌ MultiDatasetWrapper not found")
            except:
                evidence.append("❌ Could not read training script")

        score = (components / 4) * 100
        threshold_met = score >= 75

        return ValidationResult(
            criterion="Training Pipeline",
            category="Technical",
            required_value="Complete training infrastructure",
            actual_value=f"{components}/4 components",
            threshold_met=threshold_met,
            score=score,
            details="Training, evaluation, and dataset components",
            evidence=evidence
        )

    def _validate_academic_tools(self) -> ValidationResult:
        """Validate academic evaluation tools"""
        evidence = []
        tools_found = 0

        academic_tools = [
            "src/utils/metrics.py",
            "src/utils/calibration_tools.py",
            "src/utils/visualization.py",
            "src/utils/experiment_utils.py"
        ]

        for tool in academic_tools:
            if (self.project_root / tool).exists():
                tools_found += 1
                evidence.append(f"✅ {tool} exists")
            else:
                evidence.append(f"❌ {tool} missing")

        score = (tools_found / len(academic_tools)) * 100
        threshold_met = score >= 75

        return ValidationResult(
            criterion="Academic Tools",
            category="Technical",
            required_value="Complete evaluation toolkit",
            actual_value=f"{tools_found}/{len(academic_tools)} tools",
            threshold_met=threshold_met,
            score=score,
            details="Metrics, calibration, visualization, and experiment management",
            evidence=evidence
        )

    def _validate_environment_setup(self) -> ValidationResult:
        """Validate environment configuration"""
        evidence = []
        score = 0

        # Check environment files
        env_files = [
            "environment.yml",
            "requirements.txt",
            "Dockerfile",
            "tools/environment_manager.py"
        ]

        env_count = 0
        for env_file in env_files:
            if (self.project_root / env_file).exists():
                env_count += 1
                evidence.append(f"✅ {env_file} exists")
            else:
                evidence.append(f"❌ {env_file} missing")

        score = (env_count / len(env_files)) * 100
        threshold_met = score >= 75

        return ValidationResult(
            criterion="Environment Setup",
            category="Technical",
            required_value="Complete environment configuration",
            actual_value=f"{env_count}/{len(env_files)} files",
            threshold_met=threshold_met,
            score=score,
            details="Environment configuration and containerization",
            evidence=evidence
        )

    def _validate_reproducibility(self) -> ValidationResult:
        """Validate reproducibility implementation"""
        evidence = []
        score = 0

        # Check experiment_utils for reproducibility features
        exp_utils_path = self.project_root / "src/utils/experiment_utils.py"
        if exp_utils_path.exists():
            try:
                with open(exp_utils_path, 'r') as f:
                    content = f.read()

                reproducibility_features = [
                    ("seed", "random.seed\\|np.random.seed\\|torch.manual_seed"),
                    ("deterministic", "deterministic\\|reproducible"),
                    ("experiment_config", "ExperimentConfig\\|ExperimentManager")
                ]

                for feature_name, pattern in reproducibility_features:
                    if any(keyword in content for keyword in pattern.split('\\|')):
                        score += 33.33
                        evidence.append(f"✅ {feature_name} implementation found")
                    else:
                        evidence.append(f"❌ {feature_name} implementation missing")

            except Exception as e:
                evidence.append(f"❌ Error reading experiment_utils.py: {e}")
        else:
            evidence.append("❌ experiment_utils.py not found")

        threshold_met = score >= 66

        return ValidationResult(
            criterion="Reproducibility Standard",
            category="Academic",
            required_value="Seed management + deterministic training + config",
            actual_value=f"{score:.1f}/100 implementation",
            threshold_met=threshold_met,
            score=score,
            details="Reproducibility features verification",
            evidence=evidence
        )

    def _validate_statistical_rigor(self) -> ValidationResult:
        """Validate statistical analysis capabilities"""
        evidence = []
        score = 0

        metrics_path = self.project_root / "src/utils/metrics.py"
        calibration_path = self.project_root / "src/utils/calibration_tools.py"

        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    content = f.read()

                statistical_features = [
                    "confidence_interval",
                    "bootstrap",
                    "significance"
                ]

                for feature in statistical_features:
                    if feature in content:
                        score += 20
                        evidence.append(f"✅ {feature} found in metrics")

            except Exception as e:
                evidence.append(f"❌ Error reading metrics.py: {e}")
        else:
            evidence.append("❌ metrics.py not found")

        if calibration_path.exists():
            score += 40
            evidence.append("✅ Calibration tools implemented")
        else:
            evidence.append("❌ Calibration tools missing")

        threshold_met = score >= 60

        return ValidationResult(
            criterion="Statistical Rigor",
            category="Academic",
            required_value="CI + significance tests + calibration",
            actual_value=f"{score}/100 coverage",
            threshold_met=threshold_met,
            score=score,
            details="Statistical analysis capabilities",
            evidence=evidence
        )

    def _validate_documentation(self) -> ValidationResult:
        """Validate documentation quality"""
        evidence = []
        score = 0

        # Check README files
        readme_files = ["README.md", "manifests/README.md", "models/README.md"]
        readme_count = sum(1 for f in readme_files if (self.project_root / f).exists())

        evidence.append(f"README files: {readme_count}/{len(readme_files)}")
        score += (readme_count / len(readme_files)) * 40

        # Check CLAUDE.md
        if (self.project_root / "CLAUDE.md").exists():
            evidence.append("✅ CLAUDE.md project documentation exists")
            score += 30
        else:
            evidence.append("❌ CLAUDE.md missing")

        # Check docstrings in Python files
        python_files = list(self.src_dir.glob("**/*.py"))
        documented_files = 0

        for py_file in python_files[:5]:  # Sample check
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                if '"""' in content and ('Args:' in content or 'Parameters:' in content):
                    documented_files += 1
            except:
                continue

        if python_files:
            doc_ratio = documented_files / min(len(python_files), 5)
            evidence.append(f"Documented files: {documented_files}/{min(len(python_files), 5)} sampled")
            score += doc_ratio * 30

        threshold_met = score >= 70

        return ValidationResult(
            criterion="Documentation Quality",
            category="Academic",
            required_value="README + API docs + user guides",
            actual_value=f"{score:.1f}/100 completeness",
            threshold_met=threshold_met,
            score=score,
            details="Documentation coverage assessment",
            evidence=evidence
        )

    def _validate_baseline_analysis(self) -> ValidationResult:
        """Validate baseline analysis capabilities"""
        evidence = []
        score = 0

        eval_file = self.project_root / "src/stage_00/evaluate_baseline.py"
        if eval_file.exists():
            score += 50
            evidence.append("✅ Evaluation script exists")

            try:
                with open(eval_file, 'r') as f:
                    content = f.read()

                analysis_features = [
                    "calibration",
                    "metrics",
                    "visualization"
                ]

                for feature in analysis_features:
                    if feature in content.lower():
                        score += 16.67
                        evidence.append(f"✅ {feature} analysis capability found")

            except Exception as e:
                evidence.append(f"❌ Error reading evaluate_baseline.py: {e}")
        else:
            evidence.append("❌ evaluate_baseline.py not found")

        threshold_met = score >= 70

        return ValidationResult(
            criterion="Baseline Analysis",
            category="Academic",
            required_value="Comprehensive evaluation pipeline",
            actual_value=f"{score:.1f}/100 capability",
            threshold_met=threshold_met,
            score=score,
            details="Baseline model analysis capabilities",
            evidence=evidence
        )

    def _validate_project_structure(self) -> ValidationResult:
        """Validate 10-stage project structure"""
        evidence = []
        stages_found = 0

        # Check stage directories
        for i in range(10):
            stage_dir = self.src_dir / f"stage_{i:02d}"
            if stage_dir.exists():
                stages_found += 1
                evidence.append(f"✅ Stage {i:02d} directory exists")
            else:
                evidence.append(f"❌ Stage {i:02d} directory missing")

        # Check utils directory
        utils_score = 0
        if (self.src_dir / "utils").exists():
            utils_files = list((self.src_dir / "utils").glob("*.py"))
            utils_score = min(len(utils_files) / 8, 1.0) * 20  # Expect ~8 utility files
            evidence.append(f"✅ Utils directory with {len(utils_files)} files")
        else:
            evidence.append("❌ Utils directory missing")

        total_score = (stages_found / 10) * 80 + utils_score
        threshold_met = total_score >= 80

        return ValidationResult(
            criterion="Project Structure",
            category="System",
            required_value="10-stage architecture + utilities",
            actual_value=f"{stages_found}/10 stages + utils",
            threshold_met=threshold_met,
            score=total_score,
            details="Project directory structure verification",
            evidence=evidence
        )

    def _validate_configuration_management(self) -> ValidationResult:
        """Validate configuration system"""
        evidence = []
        config_count = 0

        config_files = [
            "configs/datasets.json",
            "configs/training.json",
        ]

        for config_file in config_files:
            if (self.project_root / config_file).exists():
                config_count += 1
                evidence.append(f"✅ {config_file} exists")
            else:
                evidence.append(f"❌ {config_file} missing")

        score = (config_count / len(config_files)) * 100
        threshold_met = score >= 75

        return ValidationResult(
            criterion="Configuration Management",
            category="System",
            required_value="Complete configuration system",
            actual_value=f"{config_count}/{len(config_files)} configs",
            threshold_met=threshold_met,
            score=score,
            details="Configuration files and management",
            evidence=evidence
        )

    def _validate_extensibility(self) -> ValidationResult:
        """Validate system extensibility"""
        evidence = []
        score = 0

        # Check modular stage structure
        stage_dirs = list(self.src_dir.glob("stage_*"))
        if len(stage_dirs) >= 10:
            score += 40
            evidence.append(f"✅ Modular architecture ({len(stage_dirs)} stages)")
        else:
            evidence.append(f"❌ Incomplete modular architecture ({len(stage_dirs)}/10)")

        # Check utility modules
        utils_dir = self.src_dir / "utils"
        if utils_dir.exists():
            utils_files = list(utils_dir.glob("*.py"))
            if len(utils_files) >= 6:
                score += 30
                evidence.append(f"✅ Rich utility library ({len(utils_files)} modules)")
            else:
                evidence.append(f"❌ Limited utility library ({len(utils_files)}/6)")

        # Check configuration support
        if self.configs_dir.exists():
            config_files = list(self.configs_dir.glob("*.json"))
            if config_files:
                score += 30
                evidence.append(f"✅ Configuration system ({len(config_files)} configs)")
            else:
                evidence.append("❌ No configuration files")

        threshold_met = score >= 70

        return ValidationResult(
            criterion="System Extensibility",
            category="System",
            required_value="Modular + configurable + extensible",
            actual_value=f"{score}/100 extensibility",
            threshold_met=threshold_met,
            score=score,
            details="System architecture extensibility",
            evidence=evidence
        )

    def _validate_operational_readiness(self) -> ValidationResult:
        """Validate operational readiness"""
        evidence = []
        score = 0

        # Check Docker support
        if (self.project_root / "Dockerfile").exists():
            score += 30
            evidence.append("✅ Docker containerization available")
        else:
            evidence.append("❌ Docker support missing")

        # Check inference placeholder
        if (self.project_root / "src/inference").exists():
            score += 25
            evidence.append("✅ Inference module structure exists")
        else:
            evidence.append("❌ Inference module missing")

        # Check models directory
        if (self.project_root / "models").exists():
            score += 25
            evidence.append("✅ Models directory exists")
        else:
            evidence.append("❌ Models directory missing")

        # Check experiment tracking
        if (self.project_root / "experiments").exists():
            score += 20
            evidence.append("✅ Experiment tracking directory exists")
        else:
            evidence.append("❌ Experiment tracking missing")

        threshold_met = score >= 70

        return ValidationResult(
            criterion="Operational Readiness",
            category="System",
            required_value="Production-ready infrastructure",
            actual_value=f"{score}/100 readiness",
            threshold_met=threshold_met,
            score=score,
            details="Operational infrastructure assessment",
            evidence=evidence
        )

    def calculate_overall_score(self, results: Dict[str, List[ValidationResult]]) -> Tuple[float, str]:
        """Calculate overall Stage-Gate score and status"""
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

        if overall_score >= 85:
            status = "EXCELLENT - Ready for Stage 1"
        elif overall_score >= 75:
            status = "GOOD - Ready for Stage 1"
        elif overall_score >= 65:
            status = "ACCEPTABLE - Minor improvements needed"
        elif overall_score >= 50:
            status = "NEEDS IMPROVEMENT - Significant changes required"
        else:
            status = "INADEQUATE - Major rework needed"

        return overall_score, status

    def generate_recommendations(self, results: Dict[str, List[ValidationResult]]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        for category, result_list in results.items():
            for result in result_list:
                if not result.threshold_met:
                    if result.score < 50:
                        priority = "HIGH PRIORITY"
                    elif result.score < 70:
                        priority = "MEDIUM PRIORITY"
                    else:
                        priority = "LOW PRIORITY"

                    recommendations.append(f"{priority}: {result.criterion} - {result.details}")

        if not recommendations:
            recommendations.append("Excellent work! All Stage-Gate criteria met.")

        return recommendations

    def run_comprehensive_validation(self) -> StageGateReport:
        """Run complete Stage-Gate validation"""
        print(f"\\n{'='*70}")
        print("AWARE-NET STAGE 0: COMPREHENSIVE STAGE-GATE VALIDATION")
        print(f"{'='*70}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {
            'technical': self.validate_technical_gates(),
            'academic': self.validate_academic_gates(),
            'system': self.validate_system_gates()
        }

        overall_score, status = self.calculate_overall_score(results)
        recommendations = self.generate_recommendations(results)

        next_steps = []
        if overall_score >= 75:
            next_steps = [
                "Begin Stage 1 planning and design",
                "Set up SupCon training experiments",
                "Prepare multi-dataset training pipeline",
                "Archive Stage 0 deliverables"
            ]
        else:
            next_steps = [
                "Address high-priority recommendations",
                "Re-run Stage-Gate validation",
                "Implement missing critical components",
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

    def print_summary(self, report: StageGateReport):
        """Print formatted validation summary"""
        print(f"\\n{'='*70}")
        print("STAGE-GATE VALIDATION SUMMARY")
        print(f"{'='*70}")

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

            print(f"\\n{category.title()} Gates: {passed}/{total} passed (avg: {avg_score:.1f})")

            for result in results:
                status = "✅" if result.threshold_met else "❌"
                print(f"  {status} {result.criterion}: {result.score:.1f}/100")

        # Recommendations
        if report.recommendations:
            print(f"\\n{'='*40}")
            print("RECOMMENDATIONS")
            print(f"{'='*40}")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")

        # Next Steps
        if report.next_steps:
            print(f"\\n{'='*40}")
            print("NEXT STEPS")
            print(f"{'='*40}")
            for i, step in enumerate(report.next_steps, 1):
                print(f"{i}. {step}")

        # Final assessment
        print(f"\\n{'='*70}")
        if report.overall_score >= 75:
            print("🎉 STAGE-GATE STATUS: APPROVED FOR STAGE 1")
        else:
            print("⚠️  STAGE-GATE STATUS: IMPROVEMENTS REQUIRED")
        print(f"{'='*70}")

def main():
    """Main validation script"""
    import argparse

    parser = argparse.ArgumentParser(description="AWARE-NET Stage 0 Stage-Gate Validator")
    parser.add_argument('--project-root', default='.',
                       help='Root directory of AWARE-NET project')
    parser.add_argument('--output', default='stage_gate_report.json',
                       help='Output file for validation report')

    args = parser.parse_args()

    # Initialize validator
    validator = StageGateValidator(args.project_root)

    # Run comprehensive validation
    report = validator.run_comprehensive_validation()

    # Save report
    with open(args.output, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Print summary
    validator.print_summary(report)

    print(f"\\n📄 Detailed report saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if report.overall_score >= 75 else 1)

if __name__ == "__main__":
    main()
