"""
データセット読み込みモジュール
"""
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple

import jsonschema
from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    """Few-shotサンプル"""
    input: str
    output: str


class EvaluationSample(BaseModel):
    """評価サンプル"""
    input: str
    output: str


class Dataset(BaseModel):
    """評価用データセット"""
    instruction: str
    output_length: int
    metrics: List[str]
    few_shots: List[FewShotExample] = Field(default_factory=list)
    samples: List[EvaluationSample]


class DataLoader:
    """データセット読み込みクラス"""
    
    def __init__(self, dataset_path: str):
        """
        初期化
        
        Args:
            dataset_path: データセットファイルのパス
        """
        self.dataset_path = dataset_path
        
    def load(self) -> Dataset:
        """
        データセットを読み込む
        
        Returns:
            Dataset: 読み込んだデータセット
        
        Raises:
            FileNotFoundError: データセットファイルが存在しない場合
            ValueError: データセットの形式が不正な場合
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        try:
            return Dataset(**data)
        except Exception as e:
            raise ValueError(f"Invalid dataset format: {str(e)}")
    
    def get_sample_count(self) -> int:
        """サンプル数を取得"""
        dataset = self.load()
        return len(dataset.samples)
    
    def get_metrics(self) -> List[str]:
        """評価指標を取得"""
        dataset = self.load()
        return dataset.metrics
    
    def validate(self) -> bool:
        """
        データセットの形式を検証
        
        Returns:
            bool: 検証結果 (Trueなら正常)
        """
        try:
            self.load()
            return True
        except Exception:
            return False


def load_multiple_datasets(dataset_paths: List[str]) -> Dict[str, Dataset]:
    """
    複数のデータセットを読み込む
    
    Args:
        dataset_paths: データセットファイルのパスのリスト
    
    Returns:
        Dict[str, Dataset]: データセット名とデータセットのマップ
    """
    datasets = {}
    for path in dataset_paths:
        name = os.path.basename(path).split(".")[0]
        loader = DataLoader(path)
        datasets[name] = loader.load()
    return datasets
