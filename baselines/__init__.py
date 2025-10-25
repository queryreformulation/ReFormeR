"""
Query Reformulation Baselines

This package contains implementations of several query reformulation baselines:
- MuGI: Multi-Text Generation Integration
- QA-EXPAND: Multi-Question Answer Generation
- GenQR-Ensemble: Ensemble of instruction variants
- FlanQR: Instruction-based expansion using Qwen
- Query2Doc: Pseudo-document generation variants
"""

__version__ = "1.0.0"
__author__ = "Query Reformulation Research Team"

from .mugi import MuGIGenerator
from .qa_expand import QAExpandGenerator
from .genqr_ensemble import GenQREnsembleGenerator
from .flanqr import FlanQRGenerator

__all__ = [
    "MuGIGenerator",
    "QAExpandGenerator", 
    "GenQREnsembleGenerator",
    "FlanQRGenerator"
]
