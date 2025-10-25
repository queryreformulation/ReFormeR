#!/usr/bin/env python3

import json
import pandas as pd
import argparse
import os
from typing import List, Tuple, Dict
from vllm import LLM, SamplingParams

class QAExpandGenerator:
    def __init__(self, 
                 model_name: str = None,
                 num_subquestions: int = 3,
                 repeat_query_weight: int = 3,
                 env: str = "gpu",
                 debug: bool = False):
        self.num_subquestions = num_subquestions
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
            self.llm = None
        else:
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.8,
                max_model_len=8192
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = SamplingParams(
                max_tokens=1024
            )
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
    
    def _parse_json_response(self, response: str) -> dict:
        response_clean = response.strip()
        
        if '```json' in response_clean:
            parts = response_clean.split('```json')
            if len(parts) > 1:
                response_clean = parts[1].split('```')[0].strip()
        elif '```' in response_clean:
            parts = response_clean.split('```')
            if len(parts) >= 2:
                response_clean = parts[1].strip()
        
        try:
            import re
            unicode_to_ascii = [
                ('"', '"'), ('"', '"'), (''', "'"), (''', "'"),
                ('—', '-'), ('–', '-'), (' ', ' '), ('　', ' '),
                ('，', ','), ('：', ':'),
            ]
            for u, a in unicode_to_ascii:
                if u in response_clean:
                    response_clean = response_clean.replace(u, a)

            translations = {
                '\\u201c': '"', '\\u201d': '"', '\\u2018': "'", '\\u2019': "'",
                '\\u2014': '-', '\\u2013': '-', '\\u00a0': ' ', '\\uff0c': ',', '\\uff1a': ':',
            }
            for k, v in translations.items():
                if k in response_clean:
                    response_clean = response_clean.replace(k, v)

            response_clean = re.sub(r'("[^"\\]*(?:\\.[^"\\]*)?")\s*\n\s*("(?:question|answer)\d+"\s*:)', r'\1,\n\2', response_clean)
            response_clean = re.sub(r'("[^"\\]*(?:\\.[^"\\]*)?")\s+("(?:question|answer)\d+"\s*:)', r'\1, \2', response_clean)

            response_clean = re.sub(r',\s*([}\]])', r'\1', response_clean)
        except Exception:
            pass

        try:
            return json.loads(response_clean)
        except json.JSONDecodeError as e:
            try:
                import re
                pairs = re.findall(r'"(question\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', response_clean, flags=re.MULTILINE)
                if pairs:
                    data = {}
                    for key, val in pairs:
                        try:
                            unescaped = bytes(val, 'utf-8').decode('unicode_escape')
                        except Exception:
                            unescaped = val
                        data[key] = unescaped
                    if any(k.startswith('question') for k in data.keys()):
                        return data
            except Exception:
                pass
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                lines = response_clean.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    attempt = '\n'.join(lines[:i])
                    if not attempt.endswith('}'):
                        attempt = attempt.rstrip(',') + '\n}'
                    try:
                        return json.loads(attempt)
                    except:
                        continue
            raise
    
    def _generate_with_llm(self, prompt: str) -> str:
        if self.env == "local":
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
        else:
            sampling_params = SamplingParams(
                max_tokens=1024
            )
            outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()
    
    def _create_chat_prompt(self, user_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates structured JSON responses."
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
    
    def generate_subquestions(self, query: str) -> List[str]:
        prompt_content = f"""You are a helpful assistant. Based on the following query, generate {self.num_subquestions} possible related questions that someone might ask.
Format the response as a JSON object with the following structure:
{{"question1":"First question ...",
"question2":"Second question ...",
"question3":"Third question ..."}}
Only include questions that are meaningful and logically related to the query. 
Here is the query: "{query}"
"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Sub-question Generation Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        try:
            data = self._parse_json_response(response)
            subquestions = [data[f"question{i+1}"] for i in range(self.num_subquestions)]
            
            if self.debug:
                print(f"  ✅ DEBUG: Parsed Sub-questions")
                for i, q in enumerate(subquestions, 1):
                    print(f"    {i}. {q}")
                print()
            
            return subquestions
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse sub-questions JSON: {e}")
            print(f"  Raw response: {response}")
            return [query] * self.num_subquestions
    
    def generate_answers(self, subquestions: List[str]) -> List[str]:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(subquestions)])
        
        prompt_content = f"""You are a knowledgeable assistant. The user provides 3 questions in JSON format. For each question, produce a document style answer. Each answer must: Be informative regarding the question. Return all answers in JSON format with the keys answer1, answer2, and answer3. For example:
{{"answer1": "...",
"answer2": "...",
"answer3": "..."}}

Text to answer:
{questions_text}"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Answer Generation Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        try:
            data = self._parse_json_response(response)
            answers = [data[f"answer{i+1}"] for i in range(self.num_subquestions)]
            
            if self.debug:
                print(f"  ✅ DEBUG: Parsed Answers")
                for i, a in enumerate(answers, 1):
                    print(f"    {i}. {a}")
                print()
            
            return answers
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse answers JSON: {e}")
            print(f"  Raw response: {response}")
            return [""] * self.num_subquestions
    
    def filter_and_refine_answers(self, query: str, answers: List[str]) -> List[int]:
        answers_dict = {f"answer{i+1}": a for i, a in enumerate(answers)}
        combined_input = {
            "query": query,
            "answers": answers_dict
        }
        
        prompt_content = f"""You are an evaluation assistant. You have an initial query and answers provided in JSON format. Your role is to check how relevant and correct each answer is. Return only those answers that are relevant and correct to the initial query. Omit or leave blank any that are incorrect, irrelevant, or too vague. If needed, please rewrite the answer in a better way.
Return your result in JSON with the same structure:
{{"answer1": "Relevant/correct...",
"answer2": "Relevant/correct...",
"answer3": "Relevant/correct..."}}
If an answer is irrelevant, do not include it at all or leave it empty. Focus on ensuring the final JSON only contains the best content for retrieval. Here is the combined input (initial query and answers): {json.dumps(combined_input)}"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Answer Filtering & Refinement Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        print(f"Response: {response}")
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        try:
            data = self._parse_json_response(response)
            
            kept_indices: List[int] = []
            for i in range(1, self.num_subquestions + 1):
                key = f"answer{i}"
                if key in data and isinstance(data[key], str) and data[key].strip():
                    kept_indices.append(i - 1)
            
            if self.debug:
                print(f"  ✅ DEBUG: Kept indices ({len(kept_indices)} kept): {kept_indices}")
            
            return kept_indices
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse refined answers JSON: {e}")
            print(f"  Raw response: {response}")
            return list(range(len(answers)))
    
    def create_expanded_query(self, query: str, original_answers: List[str], kept_indices: List[int]) -> str:
        selected_answers: List[str] = []
        for idx in kept_indices:
            if 0 <= idx < len(original_answers):
                selected_answers.append(original_answers[idx])

        normalized_answers = []
        for answer in selected_answers:
            try:
                normalized_answers.append(self._normalize_text(answer))
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to normalize answer: {e}")
                try:
                    normalized_answers.append(str(answer))
                except:
                    continue
        
        repeated_query = [query] * self.repeat_query_weight
        
        expanded_query = ' '.join(repeated_query + normalized_answers)
        
        try:
            expanded_query = self._normalize_text(expanded_query)
        except Exception as e:
            print(f"  ⚠️ Warning: Failed final normalization: {e}")
            expanded_query = ' '.join(expanded_query.split())
        
        if self.debug:
            print(f"\n  📊 DEBUG: Expanded Query")
            print(f"  {'='*60}")
            print(f"  Formula: (Q × {self.repeat_query_weight}) + {len(selected_answers)} kept original answers")
            print(f"  Total length: {len(expanded_query)} chars")
            print(f"  Preview: {expanded_query[:300]}...")
            print(f"  {'='*60}\n")
        
        return expanded_query
    
    def _normalize_text(self, text) -> str:
        if not isinstance(text, str):
            import json
            if isinstance(text, (list, dict)):
                text = json.dumps(text, ensure_ascii=False)
            else:
                text = str(text)
        
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        text = text.replace('- ', ' ').replace('• ', ' ').replace('* ', ' ')
        text = text.replace('**', '')
        
        import re
        text = re.sub(r'\d+\.\s+', ' ', text)
        
        text = text.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
        text = text.replace('"', '').replace("'", '')
        
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        text = text.strip()
        
        return text
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str):
        results = []
        
        for i, (qid, query) in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)} (qid: {qid})")
            print(f"Query: {query}")
            
            try:
                print(f"  Generating {self.num_subquestions} sub-questions...")
                subquestions = self.generate_subquestions(query)
                print(f"  ✓ Generated {len(subquestions)} sub-questions")
                
                print(f"Sub-questions: {subquestions}")
                
                print(f"  Generating {self.num_subquestions} answers...")
                answers = self.generate_answers(subquestions)
                print(f"  ✓ Generated {len(answers)} answers")
                
                print(f"Answers: {answers}")
                
                print(f"  Filtering and refining answers...")
                kept_indices = self.filter_and_refine_answers(query, answers)
                print(f"  ✓ Kept {len(kept_indices)} answers (indices): {kept_indices}")
                
                expanded_query = self.create_expanded_query(query, answers, kept_indices)
                
                result = {
                    'qid': qid,
                }
                
                for j, subq in enumerate(subquestions, 1):
                    result[f'subquestion_{j}'] = self._normalize_text(subq)
                
                for j, ans in enumerate(answers, 1):
                    result[f'answer_{j}'] = self._normalize_text(ans)
                result['kept_indices'] = ','.join(str(i) for i in kept_indices)
                
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
    parser = argparse.ArgumentParser(description="QA-EXPAND Baseline - Query Expansion via Sub-questions")
    
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (qid, query)")
    
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    
    parser.add_argument('--model', type=str, default=None,
                       help="Model name (default: Qwen2.5-7B for GPU, Qwen2.5-1.5B for local)")
    parser.add_argument('--num_subquestions', type=int, default=3,
                       help="Number of sub-questions to generate (default: 3)")
    parser.add_argument('--repeat_query_weight', type=int, default=3,
                       help="Number of times to repeat query in expansion (default: 3)")
    parser.add_argument('--env', type=str, default="gpu", choices=["gpu", "local"],
                       help="Environment: 'gpu' for vLLM (default) or 'local' for transformers (CPU/MPS)")
    parser.add_argument('--debug', action='store_true',
                       help="Enable verbose debug logging (shows prompts, LLM responses)")
    
    args = parser.parse_args()
    
    generator = QAExpandGenerator(
        model_name=args.model,
        num_subquestions=args.num_subquestions,
        repeat_query_weight=args.repeat_query_weight,
        env=args.env,
        debug=args.debug
    )
    
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    print("\n" + "="*60)
    print("QA-EXPAND PIPELINE")
    print(f"Parameters: num_subquestions={args.num_subquestions}, repeat_query_weight={args.repeat_query_weight}")
    print("="*60)
    generator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("QA-EXPAND GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()