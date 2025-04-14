"""
LLMクライアントモジュール
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import json

import aiohttp
import requests
from tqdm import tqdm

from ..utils.config import LLMConfig
from ..data.loader import Dataset, EvaluationSample, FewShotExample


class LLMClient:
    """LLMクライアント"""
    
    def __init__(self, config: LLMConfig):
        """
        初期化
        
        Args:
            config: LLMの設定
        """
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    def format_prompt(self, 
                      instruction: str, 
                      input_text: str, 
                      few_shots: Optional[List[FewShotExample]] = None) -> List[Dict[str, str]]:
        """
        プロンプトをフォーマットする
        
        Args:
            instruction: インストラクション
            input_text: 入力テキスト
            few_shots: Few-shotサンプル
            
        Returns:
            List[Dict[str, str]]: メッセージのリスト
        """
        messages = []
        
        # Add instruction as a system message
        messages.append({"role": "system", "content": instruction})
        
        # Add few-shot examples if provided
        if few_shots:
            for example in few_shots:
                messages.append({"role": "user", "content": example.input})
                messages.append({"role": "assistant", "content": example.output})
        
        # Add the actual input
        messages.append({"role": "user", "content": input_text})
        
        return messages
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        LLMにリクエストを送る（同期版）
        
        Args:
            messages: メッセージのリスト
            
        Returns:
            str: LLMの出力
            
        Raises:
            Exception: リクエストが失敗した場合
        """
        data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data),
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Failed to call LLM after {self.config.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    async def call_llm_async(self, messages: List[Dict[str, str]]) -> str:
        """
        LLMにリクエストを送る（非同期版）
        
        Args:
            messages: メッセージのリスト
            
        Returns:
            str: LLMの出力
            
        Raises:
            Exception: リクエストが失敗した場合
        """
        data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        async with self.semaphore:  # 並列リクエスト数を制限
            for attempt in range(self.config.max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            self.config.endpoint,
                            headers={"Content-Type": "application/json"},
                            json=data,
                            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                        ) as response:
                            if response.status != 200:
                                text = await response.text()
                                raise Exception(f"HTTP error {response.status}: {text}")
                            
                            json_response = await response.json()
                            return json_response["choices"][0]["message"]["content"]
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(f"Failed to call LLM after {self.config.max_retries} attempts: {str(e)}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def generate_predictions(self, dataset: Dataset) -> List[str]:
        """
        データセットに対する予測を生成する（同期版）
        
        Args:
            dataset: データセット
            
        Returns:
            List[str]: 予測結果のリスト
        """
        predictions = []
        for sample in tqdm(dataset.samples, desc="Generating predictions"):
            messages = self.format_prompt(
                dataset.instruction, 
                sample.input, 
                dataset.few_shots
            )
            prediction = self.call_llm(messages)
            predictions.append(prediction)
        return predictions
    
    async def generate_predictions_async(self, dataset: Dataset) -> List[str]:
        """
        データセットに対する予測を生成する（非同期版）
        
        Args:
            dataset: データセット
            
        Returns:
            List[str]: 予測結果のリスト
        """
        async def process_sample(sample: EvaluationSample) -> str:
            messages = self.format_prompt(
                dataset.instruction, 
                sample.input, 
                dataset.few_shots
            )
            return await self.call_llm_async(messages)
        
        tasks = [process_sample(sample) for sample in dataset.samples]
        
        # tqdmを使用して進捗を表示
        predictions = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating predictions"
        ):
            result = await future
            predictions.append(result)
        
        # 元の順序を維持するため、タスクの完了順ではなく、サンプルの順序に合わせる
        ordered_predictions = await asyncio.gather(*tasks)
        
        return ordered_predictions
