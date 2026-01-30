import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import json
from typing import List, Dict

class LLMAgent:
    def __init__(self, model_name: str, use_lora: bool = True, use_4bit: bool = True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def generate_params(self, instruction: str, max_new_tokens: int = 512) -> List[Dict]:
        prompt = f"""<|im_start|>system
You are a sophisticated financial analyst. Your task is to analyze a natural language instruction about a market scenario and break it down into a list of mutually exclusive potential outcomes. For each outcome, provide a descriptive scenario name, its estimated probability (all probabilities should sum to 1.0), and a JSON object of quantitative SDE parameters that mathematically describe that scenario.
<|im_end|>
<|im_start|>user
Instruction: {instruction}
JSON Output:<|im_end|>
<|im_start|>assistant
```json
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
            scenarios = json.loads(json_str)
            # 验证格式
            if isinstance(scenarios, list) and all('probability' in s and 'params' in s for s in scenarios):
                 # 归一化概率
                total_prob = sum(s.get('probability', 0) for s in scenarios)
                if total_prob > 0:
                    for s in scenarios:
                        s['probability'] /= total_prob
                return scenarios
            else:
                 raise ValueError("Parsed JSON is not a valid list of scenarios.")

        except (IndexError, json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM multi-scenario output: {e}")
            print(f"Raw response: {response_text}")
            return [{"scenario": "default", "probability": 1.0, "params": {}}]