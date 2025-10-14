"""
AWARE-NET Environment Setup Tools

This module contains tools for environment management, PyTorch installation,
and environment validation.
"""

from .environment_manager import main as environment_manager
from .setup_environment import main as setup_environment

__all__ = ['environment_manager', 'setup_environment']