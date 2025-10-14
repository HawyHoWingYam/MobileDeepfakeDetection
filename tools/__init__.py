"""
AWARE-NET Tools Package

Unified tools and utilities for the AWARE-NET deepfake detection framework.

This package contains:
- data/: Data processing and manifest generation tools
- setup/: Environment setup and validation tools
- performance/: Performance profiling and benchmarking tools
- validation/: Stage validation and gate checking tools
- tests/: Test suites for all components
"""

__version__ = "1.0.0"
__author__ = "AWARE-NET Team"

# Import submodules for easy access
from . import data
from . import setup
from . import performance
from . import validation
from . import tests

__all__ = ['data', 'setup', 'performance', 'validation', 'tests']