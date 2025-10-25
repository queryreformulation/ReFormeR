

__version__ = "1.0.0"
__author__ = "Query Reformulation Research Team"

from .core.reformer import DocumentBasedReformulator
from .patterns.query_reformulation_prompts import QueryPair, ReformulationPattern
from .patterns.iterative_pattern_extraction import IterativePatternExtractor
from .prompt_manager import PromptManager

__all__ = [
    "DocumentBasedReformulator",
    "QueryPair", 
    "ReformulationPattern",
    "IterativePatternExtractor",
    "PromptManager"
]
