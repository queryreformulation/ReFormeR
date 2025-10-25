#!/usr/bin/env python3

import json
import pandas as pd
import argparse
import os
import random
from typing import List, Dict, Tuple, Set
import time

from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm


class BasePassageGenerator:
    def __init__(self, 
                 temperature: float = 1,
                 max_tokens: int = 128,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        
        print(f"Loading vLLM model: {model_name}")
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.8
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def create_passage_prompt(self, target_query: str) -> str:
        prompt = f"Write a passage that answers the following query: {target_query}."
        return prompt
    
    def generate_passage(self, target_query: str) -> str:
        prompt = self.create_passage_prompt(target_query)

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates informative passages to answer queries."},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.llm.generate(
                prompts=[chat_prompt],
                sampling_params=self.sampling_params
            )
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return f"Error: {e}"
    
    def process_target_queries(self, target_queries: List[Tuple[str, str]], output_file: str):
        results = []
        for qid, query in tqdm(target_queries, desc="Generating passages", unit="query"):
            try:
                print("Processing qid: ", qid)
                generated_passage = self.generate_passage(query)
                results.append({
                    'qid': qid,
                    'generated_passage': generated_passage
                })
            except Exception as e:
                logging.error(f"Processing error for qid={qid}: {e}")
                results.append({
                    'qid': qid,
                    'generated_passage': f'ERROR: {str(e)}'
                })
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)
        
        return results
    
    def load_target_queries_from_file(self, file_path: str) -> List[Tuple[str, str]]:
        queries = []
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'query'])
            for _, row in df.iterrows():
                qid = str(row['qid'])
                query_text = str(row['query'])
                if query_text and query_text != 'nan':
                    queries.append((qid, query_text))
        except Exception as e:
            print(f"Error loading TSV file, trying as text file: {e}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    query = line.strip()
                    if query:
                        queries.append((str(i), query))
        
        return queries


class ZeroShotPassageGenerator(BasePassageGenerator):
    def __init__(self, 
                 temperature: float = 1,
                 max_tokens: int = 128,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(temperature, max_tokens, model_name)


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Passage Generator")
    parser.add_argument('--target_queries', type=str, required=True,
                       help="Path to TSV file containing target queries with qid and query columns")
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    parser.add_argument('--temperature', type=float, default=1,
                       help="Sampling temperature (default: 1)")
    parser.add_argument('--max_tokens', type=int, default=128,
                       help="Maximum tokens to generate (default: 128)")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for generation")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    generator = ZeroShotPassageGenerator(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.model
    )
    logging.info(f"Initialized zero-shot passage generator with vLLM")
    
    target_queries = generator.load_target_queries_from_file(args.target_queries)
    logging.info(f"Loaded target queries: {len(target_queries)}")
    
    start_time = time.time()
    results = generator.process_target_queries(target_queries, args.output)
    duration = time.time() - start_time
    qps = len(target_queries) / duration if duration > 0 else 0.0
    logging.info(f"Completed generation for {len(target_queries)} queries in {duration:.2f}s ({qps:.2f} q/s). Results: {args.output}")


if __name__ == "__main__":
    main()