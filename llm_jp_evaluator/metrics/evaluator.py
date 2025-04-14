"""
評価指標モジュール
"""
from typing import Dict, List, Optional, Any, Union, Callable
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr

from ..data.loader import Dataset, EvaluationSample


class Evaluator:
    """評価指標の計算クラス"""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        初期化
        
        Args:
            metrics: 使用する評価指標のリスト
        """
        self.metrics = metrics if metrics else ["exact_match"]
        self.metric_functions = {
            "exact_match": self.exact_match,
            "char_f1": self.char_f1,
            "entity_labeling_acc": self.entity_labeling_acc,
            "pearson": self.pearson_correlation,
            "spearman": self.spearman_correlation
        }
    
    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Exact Match (EM) 精度を計算
        
        Args:
            predictions: 予測結果
            references: 正解
            
        Returns:
            float: EM精度
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            pred = pred.strip()
            ref = ref.strip()
            if pred == ref:
                correct += 1
        return correct / len(references) if references else 0
    
    def char_f1(self, predictions: List[str], references: List[str]) -> float:
        """
        文字レベルのF1スコアを計算
        
        Args:
            predictions: 予測結果
            references: 正解
            
        Returns:
            float: F1スコア
        """
        char_f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_chars = set(pred.strip())
            ref_chars = set(ref.strip())
            
            if not ref_chars:  # 正解が空文字列の場合
                if not pred_chars:  # 予測も空文字列なら正解
                    char_f1_scores.append(1.0)
                else:  # 予測が空文字列でなければ不正解
                    char_f1_scores.append(0.0)
                continue
            
            common_chars = pred_chars.intersection(ref_chars)
            
            # 適合率、再現率、F1スコアを計算
            precision = len(common_chars) / len(pred_chars) if pred_chars else 0
            recall = len(common_chars) / len(ref_chars) if ref_chars else 0
            
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            char_f1_scores.append(f1)
        
        return np.mean(char_f1_scores) if char_f1_scores else 0
    
    def entity_labeling_acc(self, predictions: List[str], references: List[str]) -> float:
        """
        エンティティラベリング精度を計算
        
        Args:
            predictions: 予測結果
            references: 正解
            
        Returns:
            float: 精度
        """
        # エンティティ抽出のパターンを定義
        pattern = r"([\w\d\s]+)\[([\w\d\s]+)\]"
        
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            # 正解からエンティティとラベルを抽出
            ref_entities = re.findall(pattern, ref)
            
            # 予測からエンティティとラベルを抽出
            pred_entities = re.findall(pattern, pred)
            
            # 正解のエンティティとラベルのペアを辞書化
            ref_dict = {entity.strip(): label.strip() for entity, label in ref_entities}
            
            # 予測のエンティティとラベルのペアを辞書化
            pred_dict = {entity.strip(): label.strip() for entity, label in pred_entities}
            
            # 共通するエンティティについて、ラベルが正しいかチェック
            for entity, ref_label in ref_dict.items():
                total += 1
                if entity in pred_dict and pred_dict[entity] == ref_label:
                    correct += 1
        
        return correct / total if total else 0
    
    def pearson_correlation(self, predictions: List[str], references: List[str]) -> float:
        """
        ピアソン相関係数を計算
        
        Args:
            predictions: 予測結果（数値に変換可能な文字列）
            references: 正解（数値に変換可能な文字列）
            
        Returns:
            float: ピアソン相関係数
        """
        try:
            # 文字列を数値に変換
            pred_nums = []
            ref_nums = []
            
            for pred, ref in zip(predictions, references):
                try:
                    pred_num = float(pred.strip())
                    ref_num = float(ref.strip())
                    pred_nums.append(pred_num)
                    ref_nums.append(ref_num)
                except (ValueError, TypeError):
                    continue  # 変換できない場合はスキップ
            
            if len(pred_nums) < 2:  # 相関を計算するには少なくとも2ペア必要
                return 0.0
            
            # ピアソン相関係数を計算
            corr, _ = pearsonr(pred_nums, ref_nums)
            return corr
        except Exception as e:
            print(f"Error in pearson_correlation: {str(e)}")
            return 0.0
    
    def spearman_correlation(self, predictions: List[str], references: List[str]) -> float:
        """
        スピアマン相関係数を計算
        
        Args:
            predictions: 予測結果（数値に変換可能な文字列）
            references: 正解（数値に変換可能な文字列）
            
        Returns:
            float: スピアマン相関係数
        """
        try:
            # 文字列を数値に変換
            pred_nums = []
            ref_nums = []
            
            for pred, ref in zip(predictions, references):
                try:
                    pred_num = float(pred.strip())
                    ref_num = float(ref.strip())
                    pred_nums.append(pred_num)
                    ref_nums.append(ref_num)
                except (ValueError, TypeError):
                    continue  # 変換できない場合はスキップ
            
            if len(pred_nums) < 2:  # 相関を計算するには少なくとも2ペア必要
                return 0.0
            
            # スピアマン相関係数を計算
            corr, _ = spearmanr(pred_nums, ref_nums)
            return corr
        except Exception as e:
            print(f"Error in spearman_correlation: {str(e)}")
            return 0.0
    
    def evaluate(self, predictions: List[str], dataset: Dataset) -> Dict[str, float]:
        """
        データセットに対する評価を実行
        
        Args:
            predictions: 予測結果
            dataset: データセット
            
        Returns:
            Dict[str, float]: 評価指標名と評価結果のマップ
        """
        if len(predictions) != len(dataset.samples):
            raise ValueError("Predictions and references must have the same length")
        
        references = [sample.output for sample in dataset.samples]
        
        # データセットに指定された評価指標のうち、サポートされているものだけを使用
        metrics_to_use = [m for m in dataset.metrics if m in self.metric_functions]
        
        # メトリクスが指定されていない場合は、デフォルトとしてexact_matchを使用
        if not metrics_to_use:
            metrics_to_use = ["exact_match"]
        
        results = {}
        for metric_name in metrics_to_use:
            metric_fn = self.metric_functions[metric_name]
            score = metric_fn(predictions, references)
            results[metric_name] = score
        
        return results
