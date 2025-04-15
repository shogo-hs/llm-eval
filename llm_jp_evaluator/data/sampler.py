"""
サンプリングユーティリティモジュール
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from llm_jp_evaluator.data.loader import Dataset, EvaluationSample, FewShotExample


class LLMLeaderboardSampler:
    """
    llm-leaderboardのサンプリング方法を再現するクラス
    
    このクラスは、Weights & BiasesのLLMリーダーボード（Nejumi Leaderboard 3）で
    使用されているサンプリング方法を再現します。
    """
    
    def __init__(self, seed: int = 42):
        """
        サンプラーを初期化します。
        
        Args:
            seed: 乱数シードの設定、デフォルトは42
        """
        self.seed = seed
        random.seed(seed)
    
    def sample_dataset(
        self, 
        dataset_path: Union[str, Path], 
        task_name: str, 
        subset: str = "test", 
        test_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        データセットからサンプリングを行い、サンプルのリストを返します。
        
        Args:
            dataset_path: データセットのJSONファイルパス
            task_name: タスク名（例: "jsquad", "jmmlu", "wiki_ner"）
            subset: データセットの部分集合（"test" または "dev"）
            test_mode: テストモードかどうか（1サンプルのみ返す）
        
        Returns:
            サンプリングされたサンプルのリスト
        
        Raises:
            FileNotFoundError: 指定されたパスにファイルが存在しない場合
            ValueError: サポートされていないサブセットが指定された場合
        """
        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)
            
        if not dataset_path.exists():
            raise FileNotFoundError(f"指定されたパスが見つかりません: {dataset_path}")
            
        if subset not in ["test", "dev"]:
            raise ValueError("サブセットは 'test' または 'dev' である必要があります")
        
        # ファイルを読み込む
        with open(dataset_path, encoding="utf-8") as f:
            task_data = json.load(f)
        
        # サンプル数を決定する
        num_samples = self._determine_sample_count(task_name, subset, test_mode)
        
        # サンプリングを行う
        samples = task_data["samples"][:num_samples]
        
        return samples
    
    def _determine_sample_count(self, task_name: str, subset: str, test_mode: bool) -> int:
        """
        タスク名とサブセットに基づいてサンプル数を決定します。
        
        Args:
            task_name: タスク名
            subset: データセットの部分集合（"test" または "dev"）
            test_mode: テストモードかどうか
            
        Returns:
            サンプル数
        """
        if test_mode:
            return 1
            
        if "wiki" in task_name:
            return 20 if subset == "test" else 5
        elif "mmlu" in task_name:
            return 5 if subset == "test" else 1
        else:
            return 100 if subset == "test" else 10

    def sample_jaster_datasets(
        self, 
        dataset_dir: Union[str, Path], 
        subset: str = "test", 
        test_mode: bool = False,
        num_few_shots: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        jasterデータセットディレクトリから全タスクのサンプリングを行います。
        
        Args:
            dataset_dir: jasterデータセットのディレクトリパス
            subset: データセットの部分集合（"test" または "dev"）
            test_mode: テストモードかどうか
            num_few_shots: few-shotの数
            
        Returns:
            タスク名をキー、サンプルリストを値とする辞書
        """
        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)
            
        # jasterのタスクリスト
        tasks = [
            "aio",
            "alt-e-to-j",
            "alt-j-to-e",
            "chabsa",
            "commonsensemoralja",
            "jamp",
            "janli",
            "jblimp",
            "jcola-in-domain",
            "jcola-out-of-domain",
            "jcommonsenseqa",
            "jemhopqa",
            "jnli",
            "jsem",
            "jsick",
            "jsquad",
            "kuci",
            "mawps",
            "mgsm",
            "niilc",
            "wiki_coreference",
            "wiki_dependency",
            "wiki_ner",
            "wiki_pas",
            "wiki_reading",
            "wikicorpus-e-to-j",
            "wikicorpus-j-to-e"
        ]
        
        # mmlu_enとjmmluのタスクを追加
        tasks.extend(sorted({p.stem for p in dataset_dir.glob(f"{subset}/**/mmlu_en_*.json")}))
        tasks.extend(sorted({p.stem for p in dataset_dir.glob(f"{subset}/**/jmmlu*.json") if not p.stem.endswith("Choice")}))
        
        # jmmlu_robustnessを追加（few-shotの場合のみ）
        if num_few_shots > 0:
            tasks.extend(sorted({p.stem for p in dataset_dir.glob(f"{subset}/**/jmmlu*.json") if p.stem.endswith("Choice")}))
        
        # 各タスクからサンプリング
        result = {}
        for task in tasks:
            task_path = dataset_dir / subset / f"{task}.json"
            if not task_path.exists():
                print(f"タスク {task} のパスが見つかりません: {task_path}")
                continue
                
            samples = self.sample_dataset(task_path, task, subset, test_mode)
            result[task] = samples
            
        return result
    
    def get_few_shot_examples(
        self, 
        dataset_path: Union[str, Path], 
        num_few_shots: int
    ) -> List[Dict[str, str]]:
        """
        few-shotの例を取得します。
        
        Args:
            dataset_path: データセットのJSONファイルパス
            num_few_shots: few-shotの数
            
        Returns:
            few-shotの例のリスト
        """
        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)
            
        # ファイルを読み込む
        with open(dataset_path, encoding="utf-8") as f:
            task_data = json.load(f)
            
        # few-shot用のサンプルは開発セットから取得
        few_shot_samples = task_data.get("few_shot_samples", [])
        if few_shot_samples:
            # few_shot_samplesが直接提供されている場合はそれを使用
            samples = few_shot_samples
        else:
            # dev setからランダムにサンプルを選択
            dev_file = dataset_path.parent.parent / "dev" / dataset_path.name
            if dev_file.exists():
                with open(dev_file, encoding="utf-8") as f:
                    dev_data = json.load(f)
                # devデータからサンプリング
                all_samples = dev_data["samples"]
                random.shuffle(all_samples)
                samples = all_samples[:num_few_shots]
            else:
                # dev setが存在しない場合は、現在のセットからfew-shotのために使用しないサンプルを選択
                all_samples = task_data["samples"][:]
                random.shuffle(all_samples)
                samples = all_samples[:num_few_shots]
        
        return [{"input": sample["input"], "output": sample["output"]} for sample in samples[:num_few_shots]]
    
    def create_dataset_with_sampling(
        self,
        original_dataset: Dataset,
        task_name: str,
        subset: str = "test",
        test_mode: bool = False,
        num_few_shots: Optional[int] = None
    ) -> Dataset:
        """
        既存のデータセットからサンプリングを行い、新しいDatasetオブジェクトを生成します。
        
        Args:
            original_dataset: 元のDatasetオブジェクト
            task_name: タスク名
            subset: データセットの部分集合（"test" または "dev"）
            test_mode: テストモードかどうか
            num_few_shots: few-shotの数（Noneの場合は元のfew-shotsを使用）
            
        Returns:
            サンプリングされたDatasetオブジェクト
        """
        # サンプル数を決定
        num_samples = self._determine_sample_count(task_name, subset, test_mode)
        
        # サンプリングを行う
        samples = original_dataset.samples[:num_samples]
        
        # few-shotの設定
        few_shots = original_dataset.few_shots
        if num_few_shots is not None and num_few_shots != len(original_dataset.few_shots):
            # 元のfew-shotsをランダムにサンプリング
            if num_few_shots <= len(original_dataset.few_shots):
                random.shuffle(few_shots)
                few_shots = few_shots[:num_few_shots]
            # 足りない場合はサンプルから追加
            else:
                remaining = num_few_shots - len(few_shots)
                additional_samples = [s for s in original_dataset.samples if s not in samples]
                random.shuffle(additional_samples)
                for s in additional_samples[:remaining]:
                    few_shots.append(FewShotExample(input=s.input, output=s.output))
        
        # 新しいDatasetオブジェクトを作成
        return Dataset(
            instruction=original_dataset.instruction,
            output_length=original_dataset.output_length,
            metrics=original_dataset.metrics,
            few_shots=few_shots,
            samples=samples
        )

    def sample_mt_bench(
        self,
        mt_bench_path: Union[str, Path],
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        MT-benchからサンプリングを行います。
        
        Args:
            mt_bench_path: MT-benchのJSONLファイルパス
            category: サンプリングするカテゴリ（指定しない場合は全カテゴリ）
            
        Returns:
            サンプリングされた質問のリスト
        """
        if not isinstance(mt_bench_path, Path):
            mt_bench_path = Path(mt_bench_path)
            
        # MT-benchの質問を読み込む
        questions = []
        with open(mt_bench_path, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.load(line))
                
        # カテゴリでフィルタリング
        if category:
            questions = [q for q in questions if q["category"] == category]
                
        return questions
    
    def get_mt_bench_temperature(self, category: str) -> float:
        """
        MT-benchのカテゴリに応じた温度パラメータを返します。
        
        Args:
            category: MT-benchのカテゴリ
            
        Returns:
            温度パラメータ値
        """
        # カテゴリごとの温度設定
        temperature_override = {
            "writing": 0.7,
            "roleplay": 0.7,
            "extraction": 0.0,
            "math": 0.0,
            "coding": 0.0,
            "reasoning": 0.0,
            "stem": 0.1,
            "humanities": 0.1
        }
        
        return temperature_override.get(category, 0.1)


def sample_dataset_with_leaderboard_rules(
    dataset_path: Union[str, Path],
    task_name: str,
    subset: str = "test",
    test_mode: bool = False,
    num_few_shots: Optional[int] = None,
    seed: int = 42
) -> Dataset:
    """
    LLM Leaderboardのルールに基づいてデータセットをサンプリングします。
    
    Args:
        dataset_path: データセットのJSONファイルパス
        task_name: タスク名
        subset: データセットの部分集合（"test" または "dev"）
        test_mode: テストモードかどうか
        num_few_shots: few-shotの数
        seed: 乱数シード
        
    Returns:
        サンプリングされたDatasetオブジェクト
    """
    from llm_jp_evaluator.data.loader import DataLoader
    
    # データローダーを使用して元のデータセットを読み込む
    loader = DataLoader(str(dataset_path))
    original_dataset = loader.load()
    
    # サンプラーを作成
    sampler = LLMLeaderboardSampler(seed=seed)
    
    # サンプリングを行い、新しいDatasetオブジェクトを生成
    return sampler.create_dataset_with_sampling(
        original_dataset=original_dataset,
        task_name=task_name,
        subset=subset,
        test_mode=test_mode,
        num_few_shots=num_few_shots
    )
