import json
import os
from typing import Dict, Any, List
from pathlib import Path

class PromptManager:
    def __init__(self, prompts_file: str = None):
        if prompts_file is None:
            prompts_file = Path(__file__).parent / "prompts.json"
        
        with open(prompts_file, 'r') as f:
            self.prompts = json.load(f)
    
    def get_system_prompt(self, prompt_type: str) -> str:
        return self.prompts["system_prompts"][prompt_type]
    
    def get_user_prompt(self, prompt_type: str, **kwargs) -> str:
        template = self.prompts["user_prompts"][prompt_type]["template"]
        placeholders = self.prompts["user_prompts"][prompt_type]["placeholders"]
        
        for placeholder in placeholders:
            if placeholder not in kwargs:
                raise ValueError(f"Missing required placeholder: {placeholder}")
        
        return template.format(**kwargs)
    
    def get_model_config(self, model_type: str) -> str:
        return self.prompts["models"][f"default_{model_type}_model"]
    
    def get_parameter(self, param_type: str) -> Any:
        return self.prompts["parameters"][param_type]
    
    def create_messages(self, prompt_type: str, system_prompt_type: str = None, **kwargs) -> List[Dict[str, str]]:
        messages = []
        
        if system_prompt_type:
            messages.append({
                "role": "system",
                "content": self.get_system_prompt(system_prompt_type)
            })
        
        messages.append({
            "role": "user",
            "content": self.get_user_prompt(prompt_type, **kwargs)
        })
        
        return messages
