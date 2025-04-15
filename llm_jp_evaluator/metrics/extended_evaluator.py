"""
拡張評価指標モジュール - LLM Leaderboardの評価指標を実装
"""
from typing import Dict, List, Optional, Any, Union, Callable
import re
import math

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
from fuzzywuzzy import fuzz
try:
    from sacrebleu import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

from ..data.loader import Dataset, EvaluationSample


class ExtendedEvaluator:
    """拡張評価指標の計算クラス - LLM Leaderboard再現用"""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        初期化
        
        Args:
            metrics: 使用する評価指標のリスト
        """
        self.metrics = metrics if metrics else ["exact_match"]
        
        # 基本評価指標
        self.metric_functions = {
            "exact_match": self.exact_match,
            "char_f1": self.char_f1,
            "exact_match_figure": self.exact_match_figure,
            "set_f1": self.set_f1,
            "pearson": self.pearson_correlation,
            "spearman": self.spearman_correlation,
        }
        
        # 言語固有の評価指標
        if SACREBLEU_AVAILABLE:
            self.metric_functions.update({
                "bleu_ja": self.bleu_ja,
                "bleu_en": self.bleu_en,
            })
            
        # 制御性評価
        self.controllability_functions = {
            "is_all_digit": self.is_all_digit,
            "is_one_of_ABCD": self.is_one_of_ABCD,
            "is_a_b": self.is_a_b,
            "is_0_4": self.is_0_4,
            "is_0_3": self.is_0_3,
            "is_0_1": self.is_0_1,
            "is_entailment2_format": self.is_entailment2_format,
            "is_entailment3_format": self.is_entailment3_format,
            "is_jsem_format": self.is_jsem_format,
            "is_wiki_ner_format": self.is_wiki_ner_format,
            "is_wiki_dependency_format": self.is_wiki_dependency_format,
            "is_chabsa_format": self.is_chabsa_format,
        }
        
        # タスクとサブカテゴリのマッピング
        self.task_to_subcategory = {
            "alt-e-to-j": "GLP_translation",
            "alt-j-to-e": "GLP_translation",
            "wikicorpus-e-to-j": "GLP_translation",
            "wikicorpus-j-to-e": "GLP_translation",
            "jsquad": "GLP_information_extraction",
            "mawps": "GLP_mathematical_reasoning",
            "wiki_ner": "GLP_entity_extraction",
            "wiki_coreference": "GLP_entity_extraction",
            "chabsa": "GLP_entity_extraction",
            "jcommonsenseqa": "GLP_knowledge_QA",
            "jemhopqa": "GLP_knowledge_QA",
            "jmmlu": "GLP_knowledge_QA",
            "niilc": "GLP_knowledge_QA",
            "aio": "GLP_knowledge_QA",
            "mmlu_en": "GLP_English_MMLU",
            "jnli": "GLP_semantic_analysis",
            "janli": "GLP_semantic_analysis",
            "jsem": "GLP_semantic_analysis",
            "jsick": "GLP_semantic_analysis",
            "jamp": "GLP_semantic_analysis",
            "jcola-in-domain": "GLP_syntactic_analysis",
            "jcola-out-of-domain": "GLP_syntactic_analysis",
            "jblimp": "GLP_syntactic_analysis",
            "wiki_reading": "GLP_syntactic_analysis",
            "wiki_pas": "GLP_syntactic_analysis",
            "wiki_dependency": "GLP_syntactic_analysis",
            "commonsensemoralja": "ALT_ethics_moral",
            "toxicity": "ALT_toxicity",
            "humanities": "GLP_expression",
            "roleplay": "GLP_expression",
            "writing": "GLP_expression",
            "reasoning": "GLP_reasoning",
            "math": "GLP_mathematical_reasoning",
            "mgsm": "GLP_mathematical_reasoning",
            "extraction": "GLP_entity_extraction",
            "stem": "GLP_knowledge_QA",
            "coding": "ADVANCED_programing"
        }
        
        # タスクと制御性評価のマッピング
        self.task_to_controllability = {
            "aio": None,
            "alt-e-to-j": None,
            "alt-j-to-e": None,
            "chabsa": "is_chabsa_format",
            "commonsensemoralja": "is_0_1",
            "jamp": "is_entailment3_format",
            "janli": "is_entailment2_format",
            "jblimp": "is_a_b",
            "jcola-in-domain": "is_0_1",
            "jcola-out-of-domain": "is_0_1",
            "jcommonsenseqa": "is_0_4",
            "jemhopqa": None,
            "jnli": "is_entailment3_format",
            "jsem": "is_jsem_format",
            "jsick": "is_entailment3_format",
            "jsquad": None,
            "jmmlu": "is_one_of_ABCD",
            "mmlu_en": "is_one_of_ABCD",
            "kuci": "is_0_3",
            "mawps": "is_all_digit",
            "mgsm": "is_all_digit",
            "niilc": None,
            "wiki_coreference": None,
            "wiki_dependency": "is_wiki_dependency_format",
            "wiki_ner": "is_wiki_ner_format",
            "wiki_pas": None,
            "wiki_reading": None,
            "wikicorpus-e-to-j": None,
            "wikicorpus-j-to-e": None,
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
    
    def exact_match_figure(self, predictions: List[str], references: List[str]) -> float:
        """
        数値の完全一致精度を計算
        
        Args:
            predictions: 予測結果（数値を含む文字列）
            references: 正解（数値を含む文字列）
            
        Returns:
            float: 数値一致精度
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            try:
                pred_val = float(pred.strip())
                ref_val = float(ref.strip())
                if pred_val == ref_val:
                    correct += 1
            except ValueError:
                # 数値に変換できない場合は不正解
                pass
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
            # fuzzywuzzyを使ってトークンレベルのF1スコアに近似
            similarity = fuzz.token_sort_ratio(pred.strip(), ref.strip())
            char_f1_scores.append(similarity / 100.0)
        
        return np.mean(char_f1_scores) if char_f1_scores else 0
    
    def set_f1(self, predictions: List[str], references: List[str]) -> float:
        """
        セットベースのF1スコアを計算 - 行単位で集合として扱う
        
        Args:
            predictions: 予測結果（複数行の文字列）
            references: 正解（複数行の文字列）
            
        Returns:
            float: セットF1スコア
        """
        set_f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_lines = [line.strip() for line in pred.strip().split('\n')]
            ref_lines = [line.strip() for line in ref.strip().split('\n')]
            
            # 重複を削除してセットに
            pred_set = set(pred_lines)
            ref_set = set(ref_lines)
            
            if not pred_set and not ref_set:
                set_f1_scores.append(1.0)  # 両方空の場合は完全一致
                continue
                
            if not pred_set or not ref_set:
                set_f1_scores.append(0.0)  # どちらかが空の場合はF1=0
                continue
            
            # 交差部分
            intersection = pred_set.intersection(ref_set)
            
            # 適合率、再現率、F1スコアを計算
            precision = len(intersection) / len(pred_set)
            recall = len(intersection) / len(ref_set)
            
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            set_f1_scores.append(f1)
        
        return np.mean(set_f1_scores) if set_f1_scores else 0
    
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
            
            # NaNの場合は0を返す
            if math.isnan(corr):
                return 0.0
                
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
            
            # NaNの場合は0を返す
            if math.isnan(corr):
                return 0.0
                
            return corr
            
        except Exception as e:
            print(f"Error in spearman_correlation: {str(e)}")
            return 0.0
    
    def bleu_ja(self, predictions: List[str], references: List[str]) -> float:
        """
        日本語BLEUスコアを計算
        
        Args:
            predictions: 予測結果のリスト
            references: 正解のリスト
            
        Returns:
            float: BLEUスコア (0.0-1.0の範囲)
            
        Note:
            sacrebleuライブラリを使用して計算されます。
            ライブラリがインストールされていない場合は0.0を返します。
        """
        if not SACREBLEU_AVAILABLE:
            print("Warning: sacrebleu is not installed. BLEU score calculation skipped.")
            return 0.0
            
        try:
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred = pred.strip()
                ref = ref.strip()
                
                if not ref:
                    continue  # 参照が空の場合はスキップ
                
                if not pred:
                    bleu_scores.append(0.0)
                    continue
                
                # 日本語用BLEU設定
                bleu_config = {"effective_order": True, "trg_lang": "ja"}
                
                # BLEUスコア計算
                bleu = BLEU(**bleu_config)
                score = bleu.corpus_score([pred], [[ref]]).score
                bleu_scores.append(score / 100.0)  # 0-1スケールに正規化
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
            
        except Exception as e:
            print(f"Error in bleu_ja: {str(e)}")
            return 0.0
    
    def bleu_en(self, predictions: List[str], references: List[str]) -> float:
        """
        英語BLEUスコアを計算
        
        Args:
            predictions: 予測結果のリスト
            references: 正解のリスト
            
        Returns:
            float: BLEUスコア (0.0-1.0の範囲)
            
        Note:
            sacrebleuライブラリを使用して計算されます。
            ライブラリがインストールされていない場合は0.0を返します。
        """
        if not SACREBLEU_AVAILABLE:
            print("Warning: sacrebleu is not installed. BLEU score calculation skipped.")
            return 0.0
            
        try:
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred = pred.strip()
                ref = ref.strip()
                
                if not ref:
                    continue  # 参照が空の場合はスキップ
                
                if not pred:
                    bleu_scores.append(0.0)
                    continue
                
                # 英語用BLEU設定
                bleu_config = {"effective_order": True, "trg_lang": "en"}
                
                # BLEUスコア計算
                bleu = BLEU(**bleu_config)
                score = bleu.corpus_score([pred], [[ref]]).score
                bleu_scores.append(score / 100.0)  # 0-1スケールに正規化
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
            
        except Exception as e:
            print(f"Error in bleu_en: {str(e)}")
            return 0.0
    
    # ----- 制御性評価関数 -----
    
    def is_all_digit(self, text: str) -> int:
        """
        数値として解釈可能かチェック
        
        Args:
            text: チェック対象のテキスト
            
        Returns:
            int: 数値として解釈可能なら1、そうでなければ0
        """
        try:
            float(text)
            return 1
        except ValueError:
            return 0
            
    def is_one_of_ABCD(self, text: str) -> int:
        """
        A, B, C, Dのいずれかかチェック
        
        Args:
            text: チェック対象のテキスト
            
        Returns:
            int: A, B, C, Dのいずれかなら1、そうでなければ0
        """
        return 1 if text in {"A", "B", "C", "D"} else 0
    
    def is_a_b(self, text: str) -> int:
        """a または b かチェック"""
        return 1 if text in {"a", "b"} else 0
    
    def is_0_4(self, text: str) -> int:
        """0から4のいずれかかチェック"""
        return 1 if text in {"0", "1", "2", "3", "4"} else 0
    
    def is_0_3(self, text: str) -> int:
        """0から3のいずれかかチェック"""
        return 1 if text in {"0", "1", "2", "3"} else 0
    
    def is_0_1(self, text: str) -> int:
        """0または1かチェック"""
        return 1 if text in {"0", "1"} else 0
    
    def is_entailment2_format(self, text: str) -> int:
        """entailment, non-entailment形式かチェック"""
        return 1 if text in {"entailment", "non-entailment"} else 0
    
    def is_entailment3_format(self, text: str) -> int:
        """entailment, contradiction, neutral形式かチェック"""
        return 1 if text in {"entailment", "contradiction", "neutral"} else 0
    
    def is_jsem_format(self, text: str) -> int:
        """yes, no, unknown, undef形式かチェック"""
        return 1 if text in {"yes", "no", "unknown", "undef"} else 0
    
    def is_wiki_ner_format(self, text: str) -> int:
        """wiki_nerの形式にマッチするかチェック"""
        allowed_tags = {
            "組織名",
            "人名",
            "地名",
            "固有物名",
            "日付表現",
            "時刻表現",
            "金額表現",
            "割合表現",
        }
        pattern = re.compile(r"^(.+?)\\（(" + "|".join(allowed_tags) + r")\\）$")
        segments = text.split()
        for segment in segments:
            if not pattern.match(segment):
                return 0
        return 1
    
    def is_wiki_dependency_format(self, text: str) -> int:
        """wiki_dependencyの形式にマッチするかチェック"""
        pattern = re.compile(r"^.+\s*->\s*.+$")
        lines = text.split("\n")
        for line in lines:
            if not pattern.match(line):
                return 0
        return 1
    
    def is_chabsa_format(self, text: str) -> int:
        """chabsaの形式にマッチするかチェック"""
        pattern = re.compile(r"(\w+)\s+(positive|neutral|negative)")
        lines = text.split("\n")
        for line in lines:
            if not pattern.match(line):
                return 0
        return 1
    
    def check_controllability(self, predictions: List[str], task_name: str) -> float:
        """
        制御性評価を実行
        
        特定のタスクに対して出力形式が適切かどうかを評価します。
        例えば、選択肢問題で「A」「B」「C」「D」の中から選ぶべきところで
        適切な形式で回答しているかなどを評価します。
        
        Args:
            predictions: 予測結果のリスト
            task_name: タスク名（例: "jmmlu", "jsquad", "wiki_ner"）
            
        Returns:
            float: 制御性スコア（0.0〜1.0）
        """
        controllability_fn = self.task_to_controllability.get(task_name)
        if controllability_fn is None:
            return 1.0  # 制御性チェックが不要なタスクは満点
            
        controllability_fn = self.controllability_functions.get(controllability_fn)
        if controllability_fn is None:
            return 1.0  # 該当する制御性チェック関数がない場合は満点
            
        # 各予測に対して制御性を評価
        scores = [controllability_fn(pred.strip()) for pred in predictions]
        scores = [s for s in scores if s is not None]  # Noneをフィルタ
        
        return np.mean(scores) if scores else 1.0
    
    def evaluate(
        self, 
        predictions: List[str], 
        dataset: Dataset,
        task_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        データセットに対する評価を実行します
        
        LLM Leaderboardの評価方法に基づいて、生成された予測と正解データを比較し、
        各評価指標のスコアを計算します。task_nameが指定されている場合は制御性評価も行います。
        
        Args:
            predictions: 予測結果のリスト
            dataset: データセットオブジェクト
            task_name: タスク名（制御性評価に使用）
            
        Returns:
            Dict[str, float]: 評価指標名と評価結果のマップ
            
        Raises:
            ValueError: 予測と参照の長さが一致しない場合
        """
        """
        データセットに対する評価を実行
        
        Args:
            predictions: 予測結果
            dataset: データセット
            task_name: タスク名（制御性評価に使用）
            
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
        
        # 制御性評価（タスク名が指定されている場合）
        if task_name:
            controllability_score = self.check_controllability(predictions, task_name)
            results["controllability"] = controllability_score
        
        return results
