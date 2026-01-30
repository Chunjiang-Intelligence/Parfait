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
        elif provider == "ollama":
            self.client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            self.model = model or "llama3"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_instruction(self, labels: Dict) -> str:
        prompt = f"""
        你是一个资深策略分析师。请根据以下给出的未来20天行情特征（基于SDE参数），反推一段可能的市场宏观背景描述或新闻快讯。
        
        特征如下：
        - 年化漂移率 (Drift): {labels['annualized_drift']:.2f}
        - 年化波动率 (Volatility): {labels['realized_volatility']:.2f}
        - 赫斯特指数 (Hurst): {labels['hurst_exponent']:.2f} (注: >0.5表示趋势强, <0.5表示均值回归)
        - 跳跃强度 (Jump Intensity): {labels['jump_intensity']:.2f}
        - 波动率聚类系数 (GARCH Beta): {labels['garch_beta']:.2f}
        - 最大回撤 (Max Drawdown): {labels['max_drawdown']:.2f}
        
        要求：
        1. 描述要像真实的财经新闻或研报，不要提及具体的数值。
        2. 涵盖宏观事件猜测（如降息、政策突发、行业利空等）。
        3. 长度控制在100字以内。
        4. 只输出指令描述，不要有任何开场白。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Distillation API Error: {e}")
            return f"市场表现为 drift:{labels['annualized_drift']:.2f}, vol:{labels['realized_volatility']:.2f}"

    def batch_distill(self, dataset: List[Dict], limit=114514) -> List[Dict]:
        distilled_data = []
        for i, item in enumerate(dataset[:limit]):
            print(f"Distilling item {i+1}/{limit}...")
            instruction = self.generate_instruction(item['labels'])
            
            target_output = [{
                "scenario": "Historical Realization",
                "probability": 1.0,
                "params": item['labels']
            }]
            
            distilled_data.append({
                "instruction": instruction,
                "output": f"```json\n{json.dumps(target_output, indent=2)}\n```"
            })
            time.sleep(0.1)
            
        return distilled_data