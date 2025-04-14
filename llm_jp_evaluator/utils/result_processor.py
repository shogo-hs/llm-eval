"""
結果処理モジュール
"""
import os
import json
from typing import Dict, List, Optional, Any, Union
import datetime

from ..data.loader import Dataset, EvaluationSample
from ..utils.config import OutputConfig


class ResultProcessor:
    """評価結果の処理クラス"""
    
    def __init__(self, config: OutputConfig):
        """
        初期化
        
        Args:
            config: 出力設定
        """
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def save_metrics(self, metrics: Dict[str, float], dataset_name: str) -> None:
        """
        評価指標の結果を保存
        
        Args:
            metrics: 評価指標の結果
            dataset_name: データセット名
        """
        if not self.config.save_metrics:
            return
        
        result_path = os.path.join(self.config.output_dir, self.config.result_file_name)
        
        # 既存の結果がある場合は読み込む
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = {}
        
        # 現在の日時を記録
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # データセットごとの結果を保存
        if dataset_name not in results:
            results[dataset_name] = {}
        
        # 結果を更新
        results[dataset_name]["metrics"] = metrics
        results[dataset_name]["timestamp"] = timestamp
        
        # 結果を保存
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def save_predictions(self, predictions: List[str], dataset: Dataset, dataset_name: str) -> None:
        """
        予測結果を保存
        
        Args:
            predictions: 予測結果
            dataset: データセット
            dataset_name: データセット名
        """
        if not self.config.save_predictions:
            return
        
        # 予測と正解のペアを作成
        prediction_data = []
        for pred, sample in zip(predictions, dataset.samples):
            prediction_data.append({
                "input": sample.input,
                "expected": sample.output,
                "prediction": pred
            })
        
        # 予測結果を保存
        prediction_path = os.path.join(
            self.config.output_dir, 
            f"{dataset_name}_{self.config.prediction_file_name}"
        )
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
    
    def print_summary(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        評価結果のサマリーを表示
        
        Args:
            metrics: データセットごとの評価指標結果
        """
        print("-" * 60)
        print("Evaluation Results:")
        print("-" * 60)
        
        for dataset_name, dataset_metrics in metrics.items():
            print(f"Dataset: {dataset_name}")
            for metric_name, score in dataset_metrics["metrics"].items():
                print(f"  {metric_name}: {score:.4f}")
            print()
        
        # 平均スコアの計算（同じ評価指標を持つデータセット間で）
        all_metrics = set()
        for dataset_metrics in metrics.values():
            all_metrics.update(dataset_metrics["metrics"].keys())
        
        if len(metrics) > 1:  # 複数のデータセットがある場合のみ平均を計算
            print("Average Scores:")
            for metric in all_metrics:
                scores = []
                for dataset_metrics in metrics.values():
                    if metric in dataset_metrics["metrics"]:
                        scores.append(dataset_metrics["metrics"][metric])
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"  Average {metric}: {avg_score:.4f}")
            
        print("-" * 60)
