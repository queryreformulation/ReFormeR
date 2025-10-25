#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from typing import List, Tuple

class FlanQRGenerator:
    def __init__(self, 
                 model_name: str = None,
                 repeat_query_weight: int = 5,
                 env: str = "gpu",
                 debug: bool = False):

        self.repeat_query_weight = repeat_query_weight
        self.env = env
        self.debug = debug

        if model_name is None:
            model_name = "Qwen/Qwen2.5-7B-Instruct" if env == "gpu" else "Qwen/Qwen2.5-1.5B-Instruct"
        

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
            self.llm = None  # Not using vLLM
        else:
            # Use vLLM for GPU
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.7,
                max_model_len=8192
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.model = None  # Not using transformers
            self.device = "cuda"
    
    def _get_device(self):
        """Detect best available device"""
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

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates comma-separated keywords."
                },
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            

            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            return generated_text
        else:

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates comma-separated keywords."
                },
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=64,
                temperature=0.7,
                top_p=0.9 
            )
            outputs = self.llm.generate(prompts=[chat_prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()
    
    def create_flanqr_prompt(self, query: str) -> str:
        prompt = f"Improve the search effectiveness by suggesting expansion terms for the query : {query}."
        return prompt
    
    def generate_expansion(self, query: str) -> str:
        prompt = self.create_flanqr_prompt(query)
        
        if self.debug:
            print(f"\n  📝 DEBUG: FlanQR Prompt")
            print(f"  {'='*60}")
            print(prompt)
            print(f"  {'='*60}\n")
        
        expansion_text = self._generate_with_llm(prompt)
        
        if self.debug:
            print(f"  ✅ DEBUG: Generated Expansion Text (raw)")
            print(f"  {'='*60}")
            print(expansion_text)
            print(f"  {'='*60}\n")

        expansion_text = self._normalize_expansion(expansion_text)
        
        if self.debug:
            print(f"  🔧 DEBUG: Normalized Expansion Text")
            print(f"  {'='*60}")
            print(expansion_text)
            print(f"  {'='*60}\n")
        
        return expansion_text
    
    def _normalize_expansion(self, text: str) -> str:

        text = text.replace('\n', ' ').replace('\r', ' ')
        
        text = text.replace('- ', ' ').replace('• ', ' ').replace('* ', ' ')

        while '  ' in text:
            text = text.replace('  ', ' ')
        
     
        text = text.strip()
        
        return text
    
    def create_expanded_query(self, query: str, expansion_text: str) -> str:


        repeated_query = [query] * self.repeat_query_weight
        

        expanded_query = ' '.join(repeated_query) + ' ' + expansion_text

        expanded_query = expanded_query.replace('\n', ' ').replace('\r', ' ')

        while '  ' in expanded_query:
            expanded_query = expanded_query.replace('  ', ' ')
        
        expanded_query = expanded_query.strip()
        
        if self.debug:
            print(f"\n  📊 DEBUG: Expanded Query")
            print(f"  {'='*60}")
            print(f"  Formula: (Q × {self.repeat_query_weight}) + expansion text")
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
                print(f"  Generating expansion text with Flan-T5...")
                expansion_text = self.generate_expansion(query)
                print(f"  ✓ Generated expansion ({len(expansion_text)} chars)")

                expanded_query = self.create_expanded_query(query, expansion_text)

                result = {
                    'qid': qid,
                    'expansion_text': expansion_text,
                    'expanded_query': expanded_query
                }
                
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
    parser = argparse.ArgumentParser(description="FlanQR Baseline - Query Expansion with Qwen Models")
    
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (qid, query)")
    
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    
    parser.add_argument('--model', type=str, default=None,
                       help="Model name (default: Qwen2.5-7B for GPU, Qwen2.5-1.5B for local)")
    parser.add_argument('--repeat_query_weight', type=int, default=5,
                       help="Number of times to repeat query in expansion (default: 5)")
    parser.add_argument('--env', type=str, default="gpu", choices=["gpu", "local"],
                       help="Environment: 'gpu' for vLLM (default) or 'local' for transformers (CPU/MPS)")
    parser.add_argument('--debug', action='store_true',
                       help="Enable verbose debug logging (shows prompts, LLM responses)")
    
    args = parser.parse_args()
    
    generator = FlanQRGenerator(
        model_name=args.model,
        repeat_query_weight=args.repeat_query_weight,
        env=args.env,
        debug=args.debug
    )
    
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")

    print("\n" + "="*60)
    print("FLANQR PIPELINE")
    print(f"Parameters: repeat_query_weight={args.repeat_query_weight}")
    print("="*60)
    generator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("FLANQR GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

