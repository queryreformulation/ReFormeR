from typing import List, Dict, Tuple
from dataclasses import dataclass
from .prompt_manager import PromptManager

@dataclass
class QueryPair:
    original_query: str
    reformulated_query: str
    query_id: str = ""

@dataclass
class ReformulationPattern:
    pattern_name: str
    description: str
    transformation_rule: str
    examples: List[Tuple[str, str]]

def create_iterative_pattern_prompt(query_pairs: List[QueryPair], 
                                 consolidated_patterns: List[ReformulationPattern] = None,
                                 creator_max_patterns: int = 20) -> List[Dict[str, str]]:
    prompt_manager = PromptManager()
    
    query_pairs_text = "\n".join([
        f"[{i+1}] Query ID: {pair.query_id} | Original: \"{pair.original_query}\" → Reformulated: \"{pair.reformulated_query}\""
        for i, pair in enumerate(query_pairs)
    ])
    
    consolidated_patterns_text = ""
    if consolidated_patterns:
        consolidated_patterns_text = "\n".join([
            f"- {pattern.pattern_name}: {pattern.description} (Rule: {pattern.transformation_rule})"
            for pattern in consolidated_patterns
        ])
        consolidated_patterns_text = f"\nCurrent Consolidated Patterns:\n{consolidated_patterns_text}\n"
    
    return prompt_manager.create_messages(
        prompt_type="iterative_pattern_extraction",
        system_prompt_type="pattern_extraction",
        query_pairs_text=query_pairs_text,
        consolidated_patterns_text=consolidated_patterns_text,
        initial_pattern_count=len(consolidated_patterns) if consolidated_patterns else 0,
        max_patterns=creator_max_patterns
    )