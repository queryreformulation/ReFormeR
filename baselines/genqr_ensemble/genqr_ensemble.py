#!/usr/bin/env python3

import json
import pandas as pd
import argparse
import os
from typing import List, Tuple, Dict

INSTRUCTIONS = [
    "Improve the search effectiveness by suggesting expansion terms for the query",
    "Recommend expansion terms for the query to improve search results",
    "Improve the search effectiveness by suggesting useful expansion terms for the query",
    "Maximize search utility by suggesting relevant expansion phrases for the query",
    "Enhance search efficiency by proposing valuable terms to expand the query",
    "Elevate search performance by recommending relevant expansion phrases for the query",
    "Boost the search accuracy by providing helpful expansion terms to enrich the query",
    "Increase the search efficacy by offering beneficial expansion keywords for the query",
    "Optimize search results by suggesting meaningful expansion terms to enhance the query",
    "Enhance search outcomes by recommending beneficial expansion terms to supplement the query"
]

class GenQREnsembleGenerator:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 repeat_query_weight: int = 5,
                 env: str = "gpu",
                 debug: bool = False):
        self.repeat_query_weight = repeat_query_weight
        self.env = env
        self.debug = debug
        self.instructions = INSTRUCTIONS
        
        print(f"Loading model: {model_name} (env={env})")
        if env == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = self._get_device()
            print(f"  Using device: {self.device}")
            
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            
            self.model.eval()
            self.llm = None
        else:
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.7,
                max_model_len=8192
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.model = None
            self.device = "cuda"
    
    def _get_device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _generate_with_llm(self, prompt: str) -> str:
        if self.env == "local":
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.92,
                    top_k=200,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
        else:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=256,
                top_p=0.92,
                top_k=200,
                repetition_penalty=1.2
            )
            outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()
    
    def _create_chat_prompt(self, instruction: str, query: str) -> str:
        user_content = f"""You are a helpful assistant who directly provides comma separated keywords or expansion terms.
Provide as many expansion terms or keywords as possible related to the query. And do not explain yourself.

{instruction}: {query}"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates comma-separated keywords."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return chat_prompt
    
    def generate_keywords_for_instruction(self, instruction: str, query: str, instruction_num: int) -> str:
        prompt = self._create_chat_prompt(instruction, query)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Instruction {instruction_num} Prompt")
            print(f"  {'='*60}")
            print(prompt)
            print(f"  {'='*60}\n")
        
        keywords = self._generate_with_llm(prompt)
        
        if self.debug:
            print(f"  ✅ DEBUG: Generated Keywords from Instruction {instruction_num}")
            print(f"  {'='*60}")
            print(keywords)
            print(f"  {'='*60}\n")
        
        return keywords
    
    def parse_keywords(self, keywords_text: str) -> List[str]:
        keywords = []
        
        if '\n' in keywords_text or keywords_text.strip().startswith('-'):
            for line in keywords_text.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    line = line[1:].strip()
                elif line.startswith('•'):
                    line = line[1:].strip()
                elif line.startswith('*'):
                    line = line[1:].strip()
                
                if line:
                    keywords.append(line)
        else:
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        return keywords
    
    def create_expanded_query(self, query: str, all_keyword_lists: List[str]) -> str:
        repeated_query = [query] * self.repeat_query_weight
        
        all_keywords = []
        for keywords_text in all_keyword_lists:
            keywords = self.parse_keywords(keywords_text)
            all_keywords.extend(keywords)
        
        expanded_query = ' '.join(repeated_query + all_keywords)
        
        expanded_query = expanded_query.replace('\n', ' ').replace('\r', ' ')
        
        while '  ' in expanded_query:
            expanded_query = expanded_query.replace('  ', ' ')
        
        expanded_query = expanded_query.strip()
        
        if self.debug:
            print(f"\n  📊 DEBUG: Expanded Query")
            print(f"  {'='*60}")
            print(f"  Formula: (Q × {self.repeat_query_weight}) + {len(all_keywords)} keywords from {len(all_keyword_lists)} instructions")
            print(f"  Total length: {len(expanded_query)} chars")
            print(f"  Preview: {expanded_query[:300]}...")
            print(f"  {'='*60}\n")
        
        return expanded_query
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str):
        results = []
        
        for i, (qid, query) in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)} (qid: {qid})")
            print(f"Query: {query}")
            
            try:
                print(f"  Generating keywords using {len(self.instructions)} instruction variants...")
                all_keyword_lists = []
                
                for j, instruction in enumerate(self.instructions, 1):
                    if not self.debug:
                        print(f"    Instruction {j}/10...", end=" ", flush=True)
                    else:
                        print(f"\n  🔄 Processing Instruction {j}/10")
                    
                    keywords = self.generate_keywords_for_instruction(instruction, query, j)
                    all_keyword_lists.append(keywords)
                    
                    if not self.debug:
                        print(f"✓ ({len(keywords)} chars)")
                
                print(f"  ✓ Generated keywords from all {len(self.instructions)} instructions")
                
                expanded_query = self.create_expanded_query(query, all_keyword_lists)
                
                result = {
                    'qid': qid,
                }
                
                for j, keywords_text in enumerate(all_keyword_lists, 1):
                    cleaned = keywords_text.replace('\n-', ' ').replace('\n', ' ').replace('-', '').replace('  ', ' ').strip()
                    result[f'keywords_{j}'] = cleaned
                
                result['expanded_query'] = expanded_query
                
                results.append(result)
                
                print(f"  Expanded query length: {len(expanded_query)} chars")
                print(f"  Preview: {expanded_query[:100]}...")
                
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'qid': qid,
                    'expanded_query': query,
                    'error': str(e)
                })
            
            df = pd.DataFrame(results)
            df.to_csv(output_file, sep='\t', index=False)
            print(f"  💾 Saved progress ({i}/{len(queries)} queries)")
        
        print(f"\n✅ All results saved to {output_file}")
        return results
    
    def load_queries_from_tsv(self, file_path: str) -> List[Tuple[str, str]]:
        queries = []
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'query'])
            for _, row in df.iterrows():
                qid = str(row['qid'])
                query_text = str(row['query'])
                if query_text and query_text != 'nan':
                    queries.append((qid, query_text))
        except Exception as e:
            print(f"Error loading TSV: {e}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    query = line.strip()
                    if query:
                        queries.append((str(i), query))
        
        return queries


def main():
    parser = argparse.ArgumentParser(description="GenQREnsemble Baseline - Keyword Generation Ensemble")
    
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (qid, query)")
    
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for generation")
    parser.add_argument('--repeat_query_weight', type=int, default=5,
                       help="Number of times to repeat query in expansion (default: 5)")
    parser.add_argument('--env', type=str, default="gpu", choices=["gpu", "local"],
                       help="Environment: 'gpu' for vLLM (default) or 'local' for transformers (CPU/MPS)")
    parser.add_argument('--debug', action='store_true',
                       help="Enable verbose debug logging (shows prompts, LLM responses)")
    
    args = parser.parse_args()
    
    generator = GenQREnsembleGenerator(
        model_name=args.model,
        repeat_query_weight=args.repeat_query_weight,
        env=args.env,
        debug=args.debug
    )
    
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    print("\n" + "="*60)
    print("GENQRENSEMBLE PIPELINE")
    print(f"Parameters: {len(INSTRUCTIONS)} instruction variants, repeat_query_weight={args.repeat_query_weight}")
    print("="*60)
    generator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("GENQRENSEMBLE GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()