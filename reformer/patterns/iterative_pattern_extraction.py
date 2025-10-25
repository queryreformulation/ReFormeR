#!/usr/bin/env python3

import json
import logging
import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from ollama import Client
import requests
from tqdm import tqdm

from query_reformulation_prompts import (
    QueryPair, 
    ReformulationPattern,
    create_iterative_pattern_prompt
)
from ..prompt_manager import PromptManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.is_thinking_model = self._is_thinking_model(model)
        
        self.prompt_manager = PromptManager()
        
        logger.info(f"Model: {model}, is_thinking: {self.is_thinking_model}")
        
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost")
            self.ollama_client = Client(host=ollama_host)
            models_response = self.ollama_client.list()
            available_models = [model_obj.model for model_obj in models_response['models']]
            if model not in available_models:
                logger.warning(f"Model {model} not found in Ollama. Available models: {available_models}")
            else:
                logger.info(f"Model {model} found in Ollama")
        except Exception as e:
            logger.warning(f"Could not check Ollama models: {e}")
    
    def _is_thinking_model(self, model: str) -> bool:
        thinking_models = [
            'qwq:latest', 'qwq', 'qwq:32b'
        ]
        return model.lower() in [m.lower() for m in thinking_models]
    
    def _remove_thinking_tags(self, content: str) -> str:
        if not self.is_thinking_model:
            return content.strip()
        
        import re
        match = re.search(r'</think>\s*', content, flags=re.DOTALL)
        if match:
            return content[match.end():].strip()
        else:
            return content.strip()
    
    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return self._call_ollama(messages)
    
    def _call_ollama(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            ollama_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    continue
                ollama_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            if messages and messages[0]['role'] == 'system':
                system_content = messages[0]['content']
                if ollama_messages and ollama_messages[0]['role'] == 'user':
                    ollama_messages[0]['content'] = f"{system_content}\n\n{ollama_messages[0]['content']}"
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    'temperature': 0,
                    'num_predict': 2000
                }
            )
            
            raw_content = response['message']['content']
            cleaned_content = self._remove_thinking_tags(raw_content)
            
            return {
                'choices': [{
                    'message': {
                        'content': cleaned_content
                    }
                }]
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

class IterativePatternExtractor:
    def __init__(self, data_path: str, output_dir: str = "results", 
                 model: str = "qwen2.5:72b", batch_size: int = 10, max_patterns: int = 25,
                 sample_size: int = None, random_seed: int = 42):
        self.data_path = data_path
        self.model = model
        self.batch_size = batch_size
        self.max_patterns = max_patterns
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}_{model.replace('/', '_').replace(':', '_')}"
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.consolidated_patterns: List[ReformulationPattern] = []
        self.iteration_results = []
        self.individual_patterns = []
        
        self.llm_client = LLMClient(model)
        
        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "model": model,
            "data_path": data_path,
            "batch_size": batch_size,
            "max_patterns": max_patterns,
            "sample_size": sample_size,
            "random_seed": random_seed,
            "output_dir": str(self.output_dir)
        }
    
    def load_data(self) -> List[QueryPair]:
        try:
            df = pd.read_csv(self.data_path, sep='\t', names=['qid', 'original_query', 'map_original', 'reformulated_query', 'map_reformulated'])
            logger.info(f"Loaded {len(df)} total query pairs from {self.data_path}")
            
            if self.sample_size is not None:
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)
                
                if len(df) > self.sample_size:
                    df = df.sample(n=self.sample_size, random_state=self.random_seed)
                    logger.info(f"Randomly sampled {len(df)} query pairs (seed: {self.random_seed})")
                else:
                    logger.info(f"Dataset size ({len(df)}) is smaller than requested sample size ({self.sample_size}), using all data")
            
            query_pairs = []
            for idx, row in df.iterrows():
                original_query = row['original_query']
                reformulated_query = row['reformulated_query']
                
                pair = QueryPair(
                    original_query=original_query,
                    reformulated_query=reformulated_query,
                    query_id=str(row['qid'])
                )
                query_pairs.append(pair)
            
            return query_pairs
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return self.llm_client.call(messages)
    
    def extract_patterns_from_batch(self, query_pairs: List[QueryPair], 
                                  batch_number: int) -> List[ReformulationPattern]:
        messages = create_iterative_pattern_prompt(
            query_pairs, 
            self.consolidated_patterns, 
            creator_max_patterns=self.max_patterns
        )
        
        response = self.call_llm(messages)
        
        try:
            content = response['choices'][0]['message']['content'].strip()
            
            if content.startswith('{') and content.endswith('}'):
                response_data = json.loads(content)
                
                consolidated_patterns_data = response_data.get("consolidated_patterns", [])
                new_patterns = []
                for pattern_data in consolidated_patterns_data:
                    if isinstance(pattern_data, dict):
                        pattern = ReformulationPattern(
                            pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                            description=pattern_data.get("description", ""),
                            transformation_rule=pattern_data.get("transformation_rule", ""),
                            examples=pattern_data.get("examples", [])
                        )
                        new_patterns.append(pattern)
                
                individual_patterns_data = response_data.get("individual_patterns", [])
                for individual_data in individual_patterns_data:
                    if isinstance(individual_data, dict):
                        query_id = individual_data.get("query_id", "")
                        original_query = individual_data.get("original_query", "")
                        reformulated_query = individual_data.get("reformulated_query", "")
                        
                        matched_pair = None
                        if query_id:
                            for pair in query_pairs:
                                if pair.query_id == query_id:
                                    matched_pair = pair
                                    break
                        
                        if not matched_pair:
                            for pair in query_pairs:
                                if (pair.original_query == original_query and 
                                    pair.reformulated_query == reformulated_query):
                                    matched_pair = pair
                                    break
                        
                        if matched_pair:
                            individual_data["query_id"] = matched_pair.query_id
                        else:
                            if not query_id:
                                individual_data["query_id"] = str(len(self.individual_patterns) + 1)
                            logger.warning(f"Batch {batch_number}: Could not match query pair for individual pattern, using query_id: {individual_data['query_id']}")
                        
                        self.individual_patterns.append(individual_data)
                
                if len(individual_patterns_data) != len(query_pairs):
                    logger.warning(f"Batch {batch_number}: Expected {len(query_pairs)} individual patterns, got {len(individual_patterns_data)}")
                
                return new_patterns
                
            elif content.startswith('[') and content.endswith(']'):
                patterns_data = json.loads(content)
                new_patterns = []
                for pattern_data in patterns_data:
                    if isinstance(pattern_data, dict):
                        pattern = ReformulationPattern(
                            pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                            description=pattern_data.get("description", ""),
                            transformation_rule=pattern_data.get("transformation_rule", ""),
                            examples=pattern_data.get("examples", [])
                        )
                        new_patterns.append(pattern)
                
                logger.info(f"Extracted {len(new_patterns)} patterns from batch {batch_number} (old format)")
                return new_patterns
            else:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    response_data = json.loads(json_str)
                    
                    consolidated_patterns_data = response_data.get("consolidated_patterns", [])
                    new_patterns = []
                    for pattern_data in consolidated_patterns_data:
                        if isinstance(pattern_data, dict):
                            pattern = ReformulationPattern(
                                pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                                description=pattern_data.get("description", ""),
                                transformation_rule=pattern_data.get("transformation_rule", ""),
                                examples=pattern_data.get("examples", [])
                            )
                            new_patterns.append(pattern)
                    
                    individual_patterns_data = response_data.get("individual_patterns", [])
                    for individual_data in individual_patterns_data:
                        if isinstance(individual_data, dict):
                            query_id = individual_data.get("query_id", "")
                            original_query = individual_data.get("original_query", "")
                            reformulated_query = individual_data.get("reformulated_query", "")
                            
                            matched_pair = None
                            if query_id:
                                for pair in query_pairs:
                                    if pair.query_id == query_id:
                                        matched_pair = pair
                                        break
                            
                            if not matched_pair:
                                for pair in query_pairs:
                                    if (pair.original_query == original_query and 
                                        pair.reformulated_query == reformulated_query):
                                        matched_pair = pair
                                        break
                            
                            if matched_pair:
                                individual_data["query_id"] = matched_pair.query_id
                                logger.debug(f"Matched query pair with ID: {matched_pair.query_id}")
                            else:
                                if not query_id:
                                    individual_data["query_id"] = str(len(self.individual_patterns) + 1)
                                logger.warning(f"Could not match query pair, using query_id: {individual_data['query_id']}")
                            
                            self.individual_patterns.append(individual_data)
                    
                    logger.info(f"Extracted {len(new_patterns)} consolidated patterns and {len(individual_patterns_data)} individual patterns from batch {batch_number}")
                    return new_patterns
                else:
                    logger.error(f"Could not parse response as JSON: {content}")
                    return []
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response content: {response['choices'][0]['message']['content']}")
            return []
    
    def run_iterative_extraction(self):
        logger.info("Starting iterative pattern extraction")
        
        all_query_pairs = self.load_data()
        
        num_batches = (len(all_query_pairs) + self.batch_size - 1) // self.batch_size
        
        processed_queries = 0
        with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_query_pairs))
                batch_pairs = all_query_pairs[start_idx:end_idx]
                
                pbar.set_description(f"Batch {batch_num + 1}/{num_batches} ({len(batch_pairs)} pairs)")
                
                new_patterns = self.extract_patterns_from_batch(batch_pairs, batch_num + 1)
                
                if not new_patterns:
                    logger.error(f"Batch {batch_num + 1}: No patterns extracted - check LLM response")
                
                self.consolidated_patterns = new_patterns
                
                iteration_result = {
                    "batch_number": batch_num + 1,
                    "batch_size": len(batch_pairs),
                    "new_patterns": len(new_patterns),
                    "total_patterns": len(self.consolidated_patterns),
                    "patterns": [p.pattern_name for p in new_patterns]
                }
                self.iteration_results.append(iteration_result)
                
                processed_queries += len(batch_pairs)
                
                if processed_queries % 10 == 0:
                    self.update_individual_patterns_file(processed_queries)
                
                if processed_queries % 500 == 0:
                    self.save_intermediate_results(processed_queries)
                
                pbar.update(1)
                pbar.set_postfix(patterns=len(new_patterns), total=len(self.consolidated_patterns))
        
        self.save_results()
    
    def save_intermediate_results(self, processed_queries: int):
        patterns_file = self.output_dir / f"extracted_patterns_{processed_queries:05d}_queries.json"
        patterns_data = []
        for pattern in self.consolidated_patterns:
            patterns_data.append({
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "transformation_rule": pattern.transformation_rule,
                "examples": pattern.examples
            })
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        individual_patterns_file = self.output_dir / f"individual_patterns_{processed_queries:05d}_queries.json"
        with open(individual_patterns_file, 'w') as f:
            json.dump(self.individual_patterns, f, indent=2)
        
        iterations_file = self.output_dir / f"extraction_results_{processed_queries:05d}_queries.json"
        with open(iterations_file, 'w') as f:
            json.dump(self.iteration_results, f, indent=2)
        
        logger.info(f"Intermediate results saved after processing {processed_queries} queries")
        logger.info(f"Files: {patterns_file.name}, {individual_patterns_file.name}, {iterations_file.name}")
    
    def update_individual_patterns_file(self, processed_queries: int):
        individual_patterns_file = self.output_dir / "individual_patterns_LIVE.json"
        
        live_data = {
            "last_updated": datetime.now().isoformat(),
            "processed_queries": processed_queries,
            "total_individual_patterns": len(self.individual_patterns),
            "patterns": self.individual_patterns
        }
        
        with open(individual_patterns_file, 'w') as f:
            json.dump(live_data, f, indent=2)
        
        logger.info(f"Live individual patterns updated after {processed_queries} queries ({len(self.individual_patterns)} patterns)")
    
    def save_results(self):
        patterns_file = self.output_dir / "extracted_patterns_FINAL.json"
        patterns_data = []
        for pattern in self.consolidated_patterns:
            patterns_data.append({
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "transformation_rule": pattern.transformation_rule,
                "examples": pattern.examples
            })
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        individual_patterns_file = self.output_dir / "individual_patterns_FINAL.json"
        with open(individual_patterns_file, 'w') as f:
            json.dump(self.individual_patterns, f, indent=2)
        
        individual_patterns_csv = self.output_dir / "individual_patterns_FINAL.csv"
        if self.individual_patterns:
            df_individual = pd.DataFrame(self.individual_patterns)
            required_columns = ['query_id', 'original_query', 'reformulated_query', 'applied_patterns', 'explanation']
            for col in required_columns:
                if col not in df_individual.columns:
                    df_individual[col] = ''
            if 'applied_patterns' in df_individual.columns:
                df_individual['applied_patterns'] = df_individual['applied_patterns'].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else str(x)
                )
            df_individual.to_csv(individual_patterns_csv, index=False)
        
        iterations_file = self.output_dir / "extraction_results_FINAL.json"
        with open(iterations_file, 'w') as f:
            json.dump(self.iteration_results, f, indent=2)
        
        metadata_file = self.output_dir / "experiment_metadata_FINAL.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        summary_file = self.output_dir / "experiment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"QUERY REFORMULATION PATTERN EXTRACTION EXPERIMENT SUMMARY\n")
            f.write(f"================================================================\n\n")
            
            f.write(f"EXPERIMENT DETAILS:\n")
            f.write(f"------------------\n")
            f.write(f"Experiment Name: {self.experiment_metadata['experiment_name']}\n")
            f.write(f"Timestamp: {self.experiment_metadata['timestamp']}\n")
            f.write(f"Model Used: {self.experiment_metadata['model']}\n")
            f.write(f"Data File: {self.experiment_metadata['data_path']}\n")
            f.write(f"Batch Size: {self.experiment_metadata['batch_size']}\n")
            f.write(f"Max Patterns: {self.experiment_metadata['max_patterns']}\n")
            f.write(f"Output Directory: {self.experiment_metadata['output_dir']}\n\n")
            
            f.write(f"DATA STATISTICS:\n")
            f.write(f"---------------\n")
            f.write(f"Total Query Pairs Processed: {len(self.individual_patterns)}\n")
            f.write(f"Total Batches Processed: {len(self.iteration_results)}\n")
            f.write(f"Average Batch Size: {len(self.individual_patterns) / len(self.iteration_results) if self.iteration_results else 0:.1f}\n\n")
            
            f.write(f"PATTERN STATISTICS:\n")
            f.write(f"------------------\n")
            f.write(f"Final Consolidated Patterns: {len(self.consolidated_patterns)}\n")
            f.write(f"Individual Query Patterns: {len(self.individual_patterns)}\n")
            
            if self.individual_patterns:
                pattern_counts = {}
                for individual in self.individual_patterns:
                    patterns = individual.get('applied_patterns', [])
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    elif isinstance(patterns, str):
                        for pattern in patterns.split('; '):
                            if pattern.strip():
                                pattern_counts[pattern.strip()] = pattern_counts.get(pattern.strip(), 0) + 1
                
                f.write(f"Pattern Frequency Analysis:\n")
                for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {pattern}: {count} occurrences\n")
                f.write(f"\n")
            
            f.write(f"CONSOLIDATED PATTERNS:\n")
            f.write(f"---------------------\n")
            for i, pattern in enumerate(self.consolidated_patterns, 1):
                f.write(f"{i}. {pattern.pattern_name}\n")
                f.write(f"   Description: {pattern.description}\n")
                f.write(f"   Transformation Rule: {pattern.transformation_rule}\n")
                f.write(f"   Examples: {len(pattern.examples)} examples\n")
                f.write(f"\n")
            
            if self.individual_patterns:
                f.write(f"SAMPLE INDIVIDUAL PATTERNS (First 10):\n")
                f.write(f"-------------------------------------\n")
                for i, individual in enumerate(self.individual_patterns[:10], 1):
                    f.write(f"{i}. Query ID: {individual.get('query_id', 'N/A')}\n")
                    f.write(f"   Original: {individual.get('original_query', 'N/A')[:100]}...\n")
                    f.write(f"   Reformulated: {individual.get('reformulated_query', 'N/A')[:100]}...\n")
                    f.write(f"   Applied Patterns: {individual.get('applied_patterns', [])}\n")
                    f.write(f"   Explanation: {individual.get('explanation', 'N/A')[:150]}...\n")
                    f.write(f"\n")
            
            f.write(f"OUTPUT FILES:\n")
            f.write(f"-------------\n")
            f.write(f"- {patterns_file.name}: Consolidated patterns in JSON format\n")
            f.write(f"- {individual_patterns_file.name}: Individual query patterns in JSON format\n")
            f.write(f"- {individual_patterns_csv.name}: Individual query patterns in CSV format\n")
            f.write(f"- {iterations_file.name}: Iteration-by-iteration results\n")
            f.write(f"- {metadata_file.name}: Experiment metadata and configuration\n")
            f.write(f"- {summary_file.name}: This summary file\n")

def main():
    data_path = "data/diamond_dataset.tsv"
    output_dir = "results"
    
    prompt_manager = PromptManager()
    model = prompt_manager.get_model_config("ollama")
    
    logger.info(f"Starting pattern extraction with Ollama ({model})")
    
    extractor = IterativePatternExtractor(
        data_path=data_path,
        output_dir=output_dir,
        model=model,
        batch_size=10,
        max_patterns=25,
        sample_size=10000,
        random_seed=42
    )
    
    try:
        extractor.run_iterative_extraction()
        
        logger.info(f"Pattern extraction completed! {len(extractor.consolidated_patterns)} patterns, {len(extractor.individual_patterns)} individual entries")
        logger.info(f"Results saved to: {extractor.output_dir}")
        
    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()