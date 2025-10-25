#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from typing import List, Tuple
from vllm import LLM, SamplingParams


class MuGIGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", num_docs: int = 5):
        self.num_docs = num_docs
        
        print(f"Loading vLLM model: {model_name}")
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.8,
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=1.0
        )
    
    def create_mugi_prompt(self, query: str):
        messages = [
            {
                "role": "system",
                "content": "You are PassageGenGPT, an AI capable of generating concise, informative, and clear pseudo passages on specific topics."
            },
            {
                "role": "user",
                "content": f"Generate one passage that is relevant to the following query: '{query}'. The passage should be concise, informative, and clear"
            },
            {
                "role": "assistant",
                "content": "Sure, here's a passage relevant to the query:"
            }
        ]
        
        chat_prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        return chat_prompt
    
    def generate_pseudo_documents(self, query: str) -> List[str]:
        prompt = self.create_mugi_prompt(query)
        
        pseudo_docs = []
        for _ in range(self.num_docs):
            outputs = self.llm.generate(prompts=[prompt], sampling_params=self.sampling_params)
            pseudo_docs.append(outputs[0].outputs[0].text.strip())
        
        return pseudo_docs
    
    def create_enhanced_query(self, query: str, pseudo_docs: List[str], adaptive_times: int = 5) -> str:
        gen_ref = ' '.join(pseudo_docs)
        
        repetition_times = (len(gen_ref) // len(query)) // adaptive_times
        
        enhanced_query = (query + ' ') * repetition_times + gen_ref
        enhanced_query = enhanced_query.replace('\n', ' ').replace('\r', ' ').strip()
        
        return enhanced_query
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str, adaptive_times: int = 5):
        results = []
        
        for i, (qid, query) in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)} (qid: {qid})")
            print(f"Query: {query}")
            
            try:
                pseudo_docs = self.generate_pseudo_documents(query)
                print(f"Generated {len(pseudo_docs)} pseudo-documents")
                
                enhanced_query = self.create_enhanced_query(query, pseudo_docs, adaptive_times)
                
                results.append({
                    'qid': qid,
                    'original_query': query,
                    'expanded_query': enhanced_query,
                    'pseudo_doc_1': pseudo_docs[0] if len(pseudo_docs) > 0 else '',
                    'pseudo_doc_2': pseudo_docs[1] if len(pseudo_docs) > 1 else '',
                    'pseudo_doc_3': pseudo_docs[2] if len(pseudo_docs) > 2 else '',
                    'pseudo_doc_4': pseudo_docs[3] if len(pseudo_docs) > 3 else '',
                    'pseudo_doc_5': pseudo_docs[4] if len(pseudo_docs) > 4 else '',
                })
                
                print(f"Enhanced query length: {len(enhanced_query)} chars")
                print(f"First pseudo-doc: {pseudo_docs[0][:100]}...")
                
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                results.append({
                    'qid': qid,
                    'original_query': query,
                    'enhanced_query': query,
                    'error': str(e)
                })
            
            df = pd.DataFrame(results)
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Saved progress ({i}/{len(queries)} queries)")
        
        print(f"\n? All results saved to {output_file}")
        
        return results
    
    def load_queries_from_tsv(self, file_path: str) -> List[Tuple[str, str]]:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'query'])
        queries = [(str(row['qid']), str(row['query'])) for _, row in df.iterrows() 
                   if pd.notna(row['query'])]
        return queries


def main():
    parser = argparse.ArgumentParser(description="MuGI Baseline - Pseudo-document Generation")
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to input TSV file with queries (qid, query)")
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for pseudo-document generation")
    parser.add_argument('--num_docs', type=int, default=5,
                       help="Number of pseudo-documents to generate per query (default: 5)")
    parser.add_argument('--adaptive_times', type=int, default=4,
                       help="Adaptive repetition ratio p (default: 5)")
    
    args = parser.parse_args()
    
    generator = MuGIGenerator(model_name=args.model, num_docs=args.num_docs)
    
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    print("\n" + "="*60)
    print("GENERATING MUGI ENHANCED QUERIES")
    print(f"Parameters: model={args.model}, num_docs={args.num_docs}, adaptive_times={args.adaptive_times}")
    print("="*60)
    generator.process_queries(queries, args.output, args.adaptive_times)
    
    print("\n" + "="*60)
    print("MUGI GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()