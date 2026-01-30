import os
import json
import time
from typing import Dict, List
import openai

class Distiller:
    def __init__(self, provider="openai", api_key=None, base_url=None, model=None):
        self.provider = provider
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.model = model or "gpt-5"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_multi_modal_instruction(self, real_labels: Dict) -> Dict:
        prompt = f"""
        你是一个顶级宏观策略分析师。以下是某段历史行情的真实数学参数：
        {json.dumps(real_labels, indent=2)}
        
        你的任务：
        1. 想象一个能够导致这种行情的宏观指令(Instruction)。例如：“如果央行降息但通胀超预期”。
        2. 基于这个指令，除了这个真实的行情参数（作为场景A），请再脑补1-2个逻辑上自洽的替代场景（作为场景B/C）。
        3. 给出每个场景的发生概率（总和为1.0）。
        
        输出格式必须是严格的 JSON：
        {{
          "instruction": "你的宏观指令描述",
          "output": [
            {{ "scenario": "真实场景描述", "probability": 0.7, "params": {json.dumps(real_labels)} }},
            {{ "scenario": "替代场景描述", "probability": 0.3, "params": {{ ...修改后的自洽参数... }} }}
          ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You output strict JSON."},
                          {"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0.8
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Distill Error: {e}")
            return None

    def batch_distill(self, dataset: List[Dict], limit=100) -> List[Dict]:
        distilled_data = []
        for i, item in enumerate(dataset[:limit]):
            res = self.generate_multi_modal_instruction(item['labels'])
            if res:
                distilled_data.append({
                    "instruction": res['instruction'],
                    "output": f"```json\n{json.dumps(res['output'], indent=2)}\n```"
                })
            time.sleep(0.5)
        return distilled_data