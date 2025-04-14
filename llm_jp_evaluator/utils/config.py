"""
設定管理モジュール
"""
import os
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLMの設定"""
    endpoint: str = Field(default="http://localhost:8000/v1/chat/completions")
    model: str = Field(default="smollm-tiny")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=500)
    timeout: int = Field(default=60)
    max_retries: int = Field(default=3)
    max_concurrent_requests: int = Field(default=5)


class MetricsConfig(BaseModel):
    """評価指標の設定"""
    enabled: List[str] = Field(default=["exact_match", "char_f1"])
    
    @field_validator("enabled")
    def validate_metrics(cls, v: List[str]) -> List[str]:
        """サポートされている評価指標か確認"""
        supported_metrics = {
            "exact_match", "char_f1", "entity_labeling_acc", 
            "pearson", "spearman"
        }
        for metric in v:
            if metric not in supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
        return v


class OutputConfig(BaseModel):
    """出力の設定"""
    output_dir: str = Field(default="./results")
    save_predictions: bool = Field(default=True)
    save_metrics: bool = Field(default=True)
    result_file_name: str = Field(default="eval_results.json")
    prediction_file_name: str = Field(default="predictions.json")


class Config(BaseModel):
    """全体の設定"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dataset_path: str = Field(default="./dataset.json")
    verbose: bool = Field(default=False)

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """JSONファイルから設定を読み込む"""
        with open(json_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_json(self, json_path: str) -> None:
        """設定をJSONファイルに保存"""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)
