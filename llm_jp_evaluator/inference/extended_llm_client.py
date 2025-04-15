"""
拡張版LLMクライアントモジュール - LLM Leaderboard再現用
"""
import asyncio
import time
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import json

import aiohttp
import requests
from tqdm import tqdm

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from ..utils.config import LLMConfig
from ..data.loader import Dataset, EvaluationSample, FewShotExample
from ..utils.prompt_utils import format_prompts_for_litellm


class ExtendedLLMClient:
    """拡張版LLMクライアント - LLM Leaderboard再現用"""
    
    def __init__(self, config: LLMConfig):
        """
        初期化
        
        Args:
            config: LLMの設定
        """
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # litellmが利用可能か確認
        if not LITELLM_AVAILABLE:
            print("Warning: litellm is not installed. Using default HTTP requests.")
    
    def format_prompt(self, 
                      dataset: Dataset,
                      task_name: Optional[str] = None,
                      num_few_shots: int = 0,
                      chat_template_path: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        LLM Leaderboard形式でプロンプトをフォーマットする
        
        Args:
            dataset: データセット
            task_name: タスク名
            num_few_shots: few-shotの数
            chat_template_path: チャットテンプレートファイルのパス
            
        Returns:
            List[List[Dict[str, str]]]: サンプルごとのメッセージリストのリスト
        """
        # プロンプトユーティリティを使ってフォーマット
        return format_prompts_for_litellm(
            dataset=dataset,
            task_name=task_name,
            num_few_shots=num_few_shots,
            chat_template_path=chat_template_path
        )
    
    def call_litellm(self, messages: List[Dict[str, str]]) -> str:
        """
        litellmを使ってLLMにリクエストを送る（同期版）
        
        Args:
            messages: メッセージのリスト
            
        Returns:
            str: LLMの出力
            
        Raises:
            Exception: リクエストが失敗した場合
        """
        if not LITELLM_AVAILABLE:
            # litellmがない場合は従来のHTTPリクエストにフォールバック
            return self._call_llm_http(messages)
            
        for attempt in range(self.config.max_retries):
            try:
                response = litellm.completion(
                    model=self.config.model,
                    messages=messages,
                    api_base=self.config.endpoint,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Failed to call LLM after {self.config.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    async def call_litellm_async(self, messages: List[Dict[str, str]]) -> str:
        """
        litellmを使ってLLMにリクエストを送る（非同期版）
        
        Args:
            messages: メッセージのリスト
            
        Returns:
            str: LLMの出力
            
        Raises:
            Exception: リクエストが失敗した場合
        """
        if not LITELLM_AVAILABLE:
            # litellmがない場合は従来のHTTPリクエストにフォールバック
            return await self._call_llm_http_async(messages)
            
        async with self.semaphore:  # 並列リクエスト数を制限
            for attempt in range(self.config.max_retries):
                try:
                    response = await litellm.acompletion(
                        model=self.config.model,
                        messages=messages,
                        api_base=self.config.endpoint,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.timeout
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(f"Failed to call LLM after {self.config.max_retries} attempts: {str(e)}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _call_llm_http(self, messages: List[Dict[str, str]]) -> str:
        """
        HTTPリクエストを使ってLLMにリクエストを送る（litellmが使えない場合のフォールバック）
        
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
    
    async def _call_llm_http_async(self, messages: List[Dict[str, str]]) -> str:
        """
        HTTPリクエストを使ってLLMにリクエストを送る（非同期版、litellmが使えない場合のフォールバック）
        
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
    
    def generate_predictions_leaderboard(
        self, 
        dataset: Dataset, 
        task_name: Optional[str] = None,
        num_few_shots: int = 0,
        chat_template_path: Optional[str] = None
    ) -> List[str]:
        """
        LLM Leaderboard形式でデータセットに対する予測を生成する（同期版）
        
        LLM Leaderboardで使用されているプロンプト形式とサンプリング方法を使用して
        データセットの各サンプルに対する予測を生成します。
        
        Args:
            dataset: データセット（Dataset型）
            task_name: タスク名（jsquad, wiki_ner, jmmluなど）
            num_few_shots: few-shotの数（デフォルト：0）
            chat_template_path: チャットテンプレートファイルのパス（オプション）
            
        Returns:
            List[str]: 生成された予測結果のリスト（データセットの各サンプルに対応）
            
        Note:
            内部的に以下の手順で処理します:
            1. format_promptでプロンプトをフォーマット
            2. call_litellmで予測を生成（litellmが利用可能な場合）
            3. 推論間隔が設定されている場合は待機
        """
        # プロンプトのフォーマット
        all_prompts = self.format_prompt(
            dataset=dataset,
            task_name=task_name,
            num_few_shots=num_few_shots,
            chat_template_path=chat_template_path
        )
        
        # 各サンプルに対して予測を生成
        predictions = []
        for messages in tqdm(all_prompts, desc=f"Generating predictions for {task_name}"):
            if LITELLM_AVAILABLE:
                prediction = self.call_litellm(messages)
            else:
                prediction = self._call_llm_http(messages)
            predictions.append(prediction)
            
            # 推論間隔が設定されている場合は待機
            if hasattr(self.config, 'inference_interval') and self.config.inference_interval > 0:
                time.sleep(self.config.inference_interval)
                
        return predictions
    
    async def generate_predictions_leaderboard_async(
        self, 
        dataset: Dataset, 
        task_name: Optional[str] = None,
        num_few_shots: int = 0,
        chat_template_path: Optional[str] = None
    ) -> List[str]:
        """
        LLM Leaderboard形式でデータセットに対する予測を生成する（非同期版）
        
        Args:
            dataset: データセット
            task_name: タスク名
            num_few_shots: few-shotの数
            chat_template_path: チャットテンプレートファイルのパス
            
        Returns:
            List[str]: 予測結果のリスト
        """
        # プロンプトのフォーマット
        all_prompts = self.format_prompt(
            dataset=dataset,
            task_name=task_name,
            num_few_shots=num_few_shots,
            chat_template_path=chat_template_path
        )
        
        # 各サンプルに対して非同期で予測を生成する関数
        async def process_sample(messages: List[Dict[str, str]]) -> str:
            if LITELLM_AVAILABLE:
                result = await self.call_litellm_async(messages)
            else:
                result = await self._call_llm_http_async(messages)
                
            # 推論間隔が設定されている場合は待機
            if hasattr(self.config, 'inference_interval') and self.config.inference_interval > 0:
                await asyncio.sleep(self.config.inference_interval)
                
            return result
        
        # 全サンプルに対するタスクを作成
        tasks = [process_sample(messages) for messages in all_prompts]
        
        # 進捗表示しつつ完了したタスクを取得
        completed_predictions = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Generating predictions for {task_name}"
        ):
            result = await future
            completed_predictions.append(result)
        
        # 元の順序を維持するため、asyncio.gatherで結果を取得
        ordered_predictions = await asyncio.gather(*tasks)
        
        return ordered_predictions
