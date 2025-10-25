"""
Query2Doc Baseline Implementation

Pseudo-document generation variants for query reformulation
Includes Chain-of-Thought, Few-Shot, and Zero-Shot prompting variants.
"""

from .query2doc_CoT import ZeroShotPassageGenerator as CoTPassageGenerator
from .query2doc_FS import MSMarcoPassageGenerator, BEIRPassageGenerator
from .query2doc_ZS import ZeroShotPassageGenerator

__all__ = [
    "CoTPassageGenerator",
    "MSMarcoPassageGenerator", 
    "BEIRPassageGenerator",
    "ZeroShotPassageGenerator"
]
