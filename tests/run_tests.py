#!/usr/bin/env python3
"""
AWARE-NET Stage 0 Test Runner
Comprehensive test execution with Stage-Gate validation
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_pytest_command(command: List[str], description: str) -> Dict:
    """Run a pytest command and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTest completed in {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("\nSTDERR:")
            print(result.stderr)
        
        return {
            'description': description,
            'command': ' '.join(command),
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    except subprocess.TimeoutExpired:
        print(f"\nTest timed out after 30 minutes")
        return {
            'description': description,
            'command': ' '.join(command),
            'return_code': -1,
            'duration': 1800,
            'stdout': '',
            'stderr': 'Test timed out',
            'success': False
        }
    
    except Exception as e:
        print(f"\nError running test: {e}")
        return {
            'description': description,
            'command': ' '.join(command),
            'return_code': -1,
            'duration': 0,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def run_unit_tests() -> Dict:
    """Run all unit tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/',
        '-m', 'not slow and not gpu',
        '-v',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Unit Tests")


def run_integration_tests() -> Dict:
    """Run integration tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/',
        '-m', 'integration',
        '-v',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Integration Tests")


def run_performance_tests() -> Dict:
    """Run performance benchmark tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_performance_benchmarks.py',
        '-v',
        '--tb=short',
        '-s'  # Show print statements
    ]
    
    return run_pytest_command(command, "Performance Benchmark Tests")


def run_stage_gate_tests() -> Dict:
    """Run Stage-Gate validation tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_performance_benchmarks.py::TestStageGateValidation',
        '-v',
        '--tb=short',
        '-s'
    ]
    
    return run_pytest_command(command, "Stage-Gate Validation Tests")


def run_slow_tests() -> Dict:
    """Run slow tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/',
        '-m', 'slow',
        '-v',
        '--tb=short',
        '-s'
    ]
    
    return run_pytest_command(command, "Slow Tests")


def run_calibration_tests() -> Dict:
    """Run calibration-specific tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_calibration.py',
        '-v',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Calibration Tools Tests")


def run_metrics_tests() -> Dict:
    """Run metrics-specific tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_metrics.py',
        '-v',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Academic Metrics Tests")


def run_dataset_tests() -> Dict:
    """Run dataset-specific tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_dataset_config.py',
        '-v',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Dataset Configuration Tests")


def run_model_tests() -> Dict:
    """Run model-specific tests"""
    command = [
        'python', '-m', 'pytest',
        'tests/test_baseline_model.py',
        '-v',
        '--tb=short'
    ]
    
    return run_pytest_command(command, "Baseline Model Tests")


def generate_test_report(results: List[Dict], output_file: str = "test_report.json"):
    """Generate comprehensive test report"""
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    total_duration = sum(r['duration'] for r in results)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_test_suites': total_tests,
        'successful_suites': successful_tests,
        'failed_suites': failed_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': f"{total_duration // 60:.0f}m {total_duration % 60:.0f}s"
    }
    
    report = {
        'summary': summary,
        'detailed_results': results
    }
    
    # Save JSON report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def print_test_summary(results: List[Dict]):
    """Print formatted test summary"""
    
    print(f"\n{'='*80}")
    print("AWARE-NET STAGE 0 TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    total_duration = sum(r['duration'] for r in results)
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    print(f"Total Duration: {total_duration//60:.0f}m {total_duration%60:.0f}s")
    
    print(f"\n{'Test Suite':<30} {'Status':<10} {'Duration':<10}")
    print(f"{'-'*50}")
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"{result['duration']:.1f}s"
        description = result['description'][:28] + "..." if len(result['description']) > 28 else result['description']
        print(f"{description:<30} {status:<10} {duration:<10}")
    
    if failed_tests > 0:
        print(f"\nFAILED TESTS:")
        for result in results:
            if not result['success']:
                print(f"  - {result['description']}")
                if result['stderr']:
                    print(f"    Error: {result['stderr'][:100]}...")
    
    # Stage-Gate Assessment
    print(f"\n{'='*80}")
    print("STAGE-GATE ASSESSMENT")
    print(f"{'='*80}")
    
    stage_gate_criteria = {
        'Unit Tests': any(r['description'] == 'Unit Tests' and r['success'] for r in results),
        'Performance Tests': any(r['description'] == 'Performance Benchmark Tests' and r['success'] for r in results),
        'Stage-Gate Validation': any(r['description'] == 'Stage-Gate Validation Tests' and r['success'] for r in results),
        'Calibration Tools': any(r['description'] == 'Calibration Tools Tests' and r['success'] for r in results),
        'Academic Metrics': any(r['description'] == 'Academic Metrics Tests' and r['success'] for r in results),
    }
    
    stage_gate_pass = all(stage_gate_criteria.values())
    
    for criterion, passed in stage_gate_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{criterion:<30} {status}")
    
    print(f"\nOVERALL STAGE-GATE STATUS: {'✅ READY FOR STAGE 1' if stage_gate_pass else '❌ IMPROVEMENTS NEEDED'}")
    
    return stage_gate_pass


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="AWARE-NET Stage 0 Test Runner")
    
    parser.add_argument('--suite', choices=[
        'all', 'unit', 'integration', 'performance', 'stage-gate', 'slow',
        'calibration', 'metrics', 'dataset', 'model'
    ], default='all', help='Test suite to run')
    
    parser.add_argument('--output', default='test_report.json', 
                       help='Output file for test report')
    
    parser.add_argument('--no-coverage', action='store_true',
                       help='Skip code coverage analysis')
    
    args = parser.parse_args()
    
    print(f"Starting AWARE-NET Stage 0 Test Suite")
    print(f"Test Suite: {args.suite}")
    print(f"Output Report: {args.output}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    if args.suite == 'all':
        # Run comprehensive test suite
        results.extend([
            run_unit_tests(),
            run_calibration_tests(),
            run_metrics_tests(),
            run_dataset_tests(),
            run_model_tests(),
            run_performance_tests(),
            run_stage_gate_tests()
        ])
        
    elif args.suite == 'unit':
        results.append(run_unit_tests())
        
    elif args.suite == 'integration':
        results.append(run_integration_tests())
        
    elif args.suite == 'performance':
        results.append(run_performance_tests())
        
    elif args.suite == 'stage-gate':
        results.append(run_stage_gate_tests())
        
    elif args.suite == 'slow':
        results.append(run_slow_tests())
        
    elif args.suite == 'calibration':
        results.append(run_calibration_tests())
        
    elif args.suite == 'metrics':
        results.append(run_metrics_tests())
        
    elif args.suite == 'dataset':
        results.append(run_dataset_tests())
        
    elif args.suite == 'model':
        results.append(run_model_tests())
    
    # Generate and print summary
    report = generate_test_report(results, args.output)
    stage_gate_pass = print_test_summary(results)
    
    print(f"\nDetailed test report saved to: {args.output}")
    
    # Exit with appropriate code
    if all(r['success'] for r in results) and stage_gate_pass:
        print("\n🎉 All tests passed! Stage 0 is ready for completion.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the results and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()