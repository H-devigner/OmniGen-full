"""OmniGen TensorFlow Implementation

This module provides the TensorFlow implementation of OmniGen, matching the PyTorch
version's functionality while leveraging TensorFlow-specific optimizations.
"""

from .model import OmniGen
from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler
from .pipeline import OmniGenPipeline
from .transformer import Phi3Config, Phi3Transformer
from .utils import *

__version__ = "1.0.0"

__all__ = [
    # Core components
    "OmniGen",
    "OmniGenProcessor",
    "OmniGenScheduler",
    "OmniGenPipeline",
    
    # Transformer components
    "Phi3Config",
    "Phi3Transformer",
    
    # Version
    "__version__",
]