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
                 collection_path: str,
                 train_queries_path: str,
                 train_qrels_path: str,
                 temperature: float = 1,
                 max_tokens: int = 128,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.collection_path = collection_path
        self.train_queries_path = train_queries_path
        self.train_qrels_path = train_qrels_path
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
        
        self._load_collection()
        self._load_train_queries()
        self._load_train_qrels()
    
    def _load_collection(self):
        raise NotImplementedError
    
    def _load_train_queries(self):
        raise NotImplementedError
    
    def _load_train_qrels(self):
        raise NotImplementedError
    
    def get_few_shot_examples(self, target_query: str, num_examples: int = 4) -> List[Tuple[str, str]]:
        raise NotImplementedError
    
    def create_passage_prompt(self, target_query: str, few_shot_examples: List[Tuple[str, str]]) -> str:
        examples_text = ""
        for i, (query, passage) in enumerate(few_shot_examples, 1):
            examples_text += f"Query: {query}\nPassage: {passage}\n\n"
        
        prompt = f"""Write a passage that answers the given query: \n\n{examples_text}Query: {target_query}
Passage:"""
        return prompt
    
    def generate_passage(self, target_query: str, few_shot_examples: List[Tuple[str, str]]) -> str:
        prompt = self.create_passage_prompt(target_query, few_shot_examples)

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
                few_shot_examples = self.get_few_shot_examples(query, num_examples=4, target_qid=qid)
                if len(few_shot_examples) == 0:
                    results.append({
                        'qid': qid,
                        'generated_passage': 'ERROR: No few-shot examples found'
                    })
                    continue
                generated_passage = self.generate_passage(query, few_shot_examples)
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


class MSMarcoPassageGenerator(BasePassageGenerator):
    def __init__(self, 
                 collection_path: str,
                 train_queries_path: str,
                 train_qrels_path: str,
                 temperature: float = 1,
                 max_tokens: int = 128,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(collection_path, train_queries_path, train_qrels_path, temperature, max_tokens, model_name)
    
    def _load_collection(self):
        self.collection_df = pd.read_csv(self.collection_path, sep='\t', header=None,
                                        names=['doc_id', 'doc_text'])
    
    def _load_train_queries(self):
        self.train_queries_df = pd.read_csv(self.train_queries_path, sep='\t', header=None,
                                           names=['query_id', 'query_text'])
    
    def _load_train_qrels(self):
        self.train_qrels_df = pd.read_csv(self.train_qrels_path, sep='\t', header=None,
                                         names=['query_id', '0','doc_id', 'relevance'])
    
    def get_few_shot_examples(self, target_query: str, num_examples: int = 4, target_qid: str = None) -> List[Tuple[str, str]]:
        relevant_pairs = self.train_qrels_df[self.train_qrels_df['relevance'] > 0]
        
        if len(relevant_pairs) == 0:
            return []
        
        sample_size = min(num_examples * 10, len(relevant_pairs))
        sampled_pairs = relevant_pairs.sample(n=sample_size)
        
        examples = []
        for _, pair in sampled_pairs.iterrows():
            if len(examples) >= num_examples:
                break
                
            query_id = pair['query_id']
            doc_id = pair['doc_id']
            
            query_info = self.train_queries_df[self.train_queries_df['query_id'] == query_id]
            if len(query_info) == 0:
                continue
            
            query_text = query_info.iloc[0]['query_text']
            
            doc_info = self.collection_df[self.collection_df['doc_id'] == doc_id]
            if len(doc_info) == 0:
                continue
            
            doc_text = doc_info.iloc[0]['doc_text']
            
            examples.append((query_text, doc_text))
        
        return examples




def main():
    parser = argparse.ArgumentParser(description="Passage Generator for MS MARCO dataset")
    parser.add_argument('--collection', type=str, required=True,
                       help="Path to collection file (collection.tsv for MS MARCO)")
    parser.add_argument('--train_queries', type=str, required=True,
                       help="Path to train queries file (TSV for MS MARCO)")
    parser.add_argument('--train_qrels', type=str, required=True,
                       help="Path to train qrels file (TSV format)")
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
    
    generator = MSMarcoPassageGenerator(
        collection_path=args.collection,
        train_queries_path=args.train_queries,
        train_qrels_path=args.train_qrels,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.model
    )
    logging.info(f"Initialized MS MARCO passage generator with vLLM")
    
    target_queries = generator.load_target_queries_from_file(args.target_queries)
    logging.info(f"Loaded target queries: {len(target_queries)}")
    
    start_time = time.time()
    results = generator.process_target_queries(target_queries, args.output)
    duration = time.time() - start_time
    qps = len(target_queries) / duration if duration > 0 else 0.0
    logging.info(f"Completed generation for {len(target_queries)} queries in {duration:.2f}s ({qps:.2f} q/s). Results: {args.output}")


if __name__ == "__main__":
    main()