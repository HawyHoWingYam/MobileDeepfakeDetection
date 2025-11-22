"""
Mobile Deployment Utilities
==========================

Utilities for exporting optimized models to mobile deployment formats
including ONNX and TensorFlow Lite conversion.
"""

from .onnx_exporter import ONNXExporter

# Optional imports - only import if modules exist
try:
    from .tflite_converter import TFLiteConverter
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

try:
    from .mobile_inference import MobileInferenceWrapper
    MOBILE_INFERENCE_AVAILABLE = True
except ImportError:
    MOBILE_INFERENCE_AVAILABLE = False

try:
    from .deployment_validator import DeploymentValidator
    DEPLOYMENT_VALIDATOR_AVAILABLE = True
except ImportError:
    DEPLOYMENT_VALIDATOR_AVAILABLE = False

__all__ = ['ONNXExporter']

if TFLITE_AVAILABLE:
    __all__.append('TFLiteConverter')
if MOBILE_INFERENCE_AVAILABLE:
    __all__.append('MobileInferenceWrapper')
if DEPLOYMENT_VALIDATOR_AVAILABLE:
    __all__.append('DeploymentValidator')