#!/usr/bin/env python3

import json
import pandas as pd
import argparse
import os
from typing import List, Dict, Tuple
import time
import logging

from vllm import LLM, SamplingParams
from ..prompt_manager import PromptManager

class Reformulator:
    def __init__(self, 
                 bm25_run_path: str,
                 collection_path: str,
                 patterns_path: str,
                 dataset_type: str = "msmarco",
                 top_k: int = 10,
                 batch_size: int = 5,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_tokens: int = 500,
                 temperature: float = 0.1):
        self.bm25_run_path = bm25_run_path
        self.collection_path = collection_path
        self.patterns_path = patterns_path
        self.dataset_type = dataset_type
        self.top_k = top_k
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
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
        
        self.prompt_manager = PromptManager()
        
        self._load_bm25_runs()
        self._load_collection()
        self._load_patterns()
    
    def _load_bm25_runs(self):
        print("Loading BM25 run file...")
        
        if self.dataset_type == "msmarco":
            self.bm25_df = pd.read_csv(self.bm25_run_path, sep='\t', header=None,
                                      names=['query_id', 'doc_id', 'rank'])
        elif self.dataset_type == "beir":
            self.bm25_df = pd.read_csv(self.bm25_run_path, sep=' ', header=None,
                                      names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'system'])
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")
            
        print(f"Loaded {len(self.bm25_df)} BM25 run entries")
    
    def _load_collection(self):
        print(f"Loading {self.dataset_type} document collection...")
        
        if self.dataset_type == "msmarco":
            self.collection_df = pd.read_csv(self.collection_path, sep='\t', header=None,
                                            names=['doc_id', 'doc_text'])
            print(f"Loaded {len(self.collection_df)} MS MARCO documents")
        elif self.dataset_type == "beir":
            self.collection_dict = {}
            with open(self.collection_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    self.collection_dict[doc['_id']] = {
                        'title': doc.get('title', ''),
                        'text': doc.get('text', ''),
                        'metadata': doc.get('metadata', {})
                    }
            print(f"Loaded {len(self.collection_dict)} BEIR documents")
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}. Must be 'msmarco' or 'beir'")
    
    def _load_patterns(self):
        print("Loading consolidated patterns...")
        with open(self.patterns_path, 'r') as f:
            self.patterns = json.load(f)
        print(f"Loaded {len(self.patterns)} patterns")
    
    def get_top_documents(self, query_id: str) -> List[str]:
        if self.dataset_type == "msmarco":
            top_docs = self.bm25_df[self.bm25_df['query_id'] == query_id].head(self.top_k)
            doc_texts = []
            for _, row in top_docs.iterrows():
                doc_id = row['doc_id']
                doc_info = self.collection_df[self.collection_df['doc_id'] == doc_id]
                if len(doc_info) > 0:
                    doc_texts.append(doc_info.iloc[0]['doc_text'])
            return doc_texts
        elif self.dataset_type == "beir":
            top_docs = self.bm25_df[self.bm25_df['query_id'] == query_id].head(self.top_k)
            doc_texts = []
            for _, row in top_docs.iterrows():
                doc_id = row['doc_id']
                if doc_id in self.collection_dict:
                    doc_data = self.collection_dict[doc_id]
                    doc_texts.append(doc_data['text'])
            return doc_texts
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")
    
    def select_best_pattern(self, query: str, documents: List[str]) -> Dict:
        if not documents:
            return self.patterns[0] if self.patterns else None
        
        documents_text = "\n\n".join(documents[:3])
        
        try:
            messages = self.prompt_manager.create_messages(
                prompt_type="pattern_selection",
                system_prompt_type="pattern_selection",
                query=query,
                documents_text=documents_text,
                patterns_json=json.dumps(self.patterns, indent=2)
            )
            
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.llm.generate(
                prompts=[chat_prompt],
                sampling_params=self.sampling_params
            )
            response_content = outputs[0].outputs[0].text.strip()
            
            selected_pattern = json.loads(response_content)
            return selected_pattern
        except Exception as e:
            logging.error(f"Error selecting pattern: {e}")
            return self.patterns[0] if self.patterns else None
    
    def apply_pattern(self, query: str, pattern: Dict) -> str:
        if not pattern:
            return query
        
        transformation_rule = pattern.get('transformation_rule', '')
        
        try:
            messages = self.prompt_manager.create_messages(
                prompt_type="pattern_application",
                system_prompt_type="pattern_application",
                query=query,
                transformation_rule=transformation_rule,
                examples_json=json.dumps(pattern.get('examples', []), indent=2)
            )
            
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
            logging.error(f"Error applying pattern: {e}")
            return query
    
    def reformulate_query(self, query_id: str, query: str) -> Dict:
        documents = self.get_top_documents(query_id)
        selected_pattern = self.select_best_pattern(query, documents)
        reformulated_query = self.apply_pattern(query, selected_pattern)
        
        return {
            'query_id': query_id,
            'original_query': query,
            'reformulated_query': reformulated_query,
            'selected_pattern': selected_pattern.get('pattern_name', 'Unknown') if selected_pattern else 'None',
            'num_documents_used': len(documents)
        }
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str):
        results = []
        
        for i, (query_id, query) in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)} (ID: {query_id})")
            
            try:
                result = self.reformulate_query(query_id, query)
                results.append(result)
                
                print(f"  Original: {query}")
                print(f"  Reformulated: {result['reformulated_query']}")
                print(f"  Pattern: {result['selected_pattern']}")
                print()
                
            except Exception as e:
                logging.error(f"Error processing query {query_id}: {e}")
                results.append({
                    'query_id': query_id,
                    'original_query': query,
                    'reformulated_query': query,
                    'selected_pattern': 'Error',
                    'error': str(e)
                })
            
            if i % self.batch_size == 0:
                df = pd.DataFrame(results)
                df.to_csv(output_file, sep='\t', index=False)
                print(f"Saved progress: {i}/{len(queries)} queries")
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)
        print(f"All results saved to {output_file}")
        
        return results
    
    def load_queries_from_tsv(self, file_path: str) -> List[Tuple[str, str]]:
        queries = []
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'query'])
            for _, row in df.iterrows():
                query_id = str(row['query_id'])
                query_text = str(row['query'])
                if query_text and query_text != 'nan':
                    queries.append((query_id, query_text))
        except Exception as e:
            print(f"Error loading TSV: {e}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    query = line.strip()
                    if query:
                        queries.append((str(i), query))
        
        return queries


def main():
    parser = argparse.ArgumentParser(description="Document-Based Query Reformulation")
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (query_id, query)")
    parser.add_argument('--bm25_run', type=str, required=True,
                       help="Path to BM25 run file")
    parser.add_argument('--collection', type=str, required=True,
                       help="Path to document collection file")
    parser.add_argument('--patterns', type=str, required=True,
                       help="Path to patterns JSON file")
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    parser.add_argument('--dataset_type', type=str, default="msmarco", choices=["msmarco", "beir"],
                       help="Dataset type: msmarco or beir")
    parser.add_argument('--top_k', type=int, default=10,
                       help="Number of top documents to retrieve")
    parser.add_argument('--batch_size', type=int, default=5,
                       help="Batch size for processing")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for generation")
    parser.add_argument('--temperature', type=float, default=0.1,
                       help="Sampling temperature (default: 0.1)")
    parser.add_argument('--max_tokens', type=int, default=500,
                       help="Maximum tokens to generate (default: 500)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    reformulator = Reformulator(
        bm25_run_path=args.bm25_run,
        collection_path=args.collection,
        patterns_path=args.patterns,
        dataset_type=args.dataset_type,
        top_k=args.top_k,
        batch_size=args.batch_size,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"Loading queries from: {args.queries}")
    queries = reformulator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    print("\n" + "="*60)
    print("DOCUMENT-BASED QUERY REFORMULATION")
    print("="*60)
    reformulator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("REFORMULATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()