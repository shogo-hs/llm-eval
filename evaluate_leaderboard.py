#!/usr/bin/env python
"""
LLM Leaderboard形式で評価を行うスクリプト
"""
import os
import sys
import json
import argparse
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from llm_jp_evaluator.data.loader import DataLoader, load_dataset_with_sampling
from llm_jp_evaluator.inference.extended_llm_client import ExtendedLLMClient
from llm_jp_evaluator.metrics.extended_evaluator import ExtendedEvaluator
from llm_jp_evaluator.utils.config import Config, LLMConfig
from llm_jp_evaluator.utils.result_processor import ResultProcessor


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパースする
    
    Returns:
        argparse.Namespace: パースされたコマンドライン引数
    """
    parser = argparse.ArgumentParser(description="Evaluate LLMs using Nejumi Leaderboard methods")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+", 
        help="Path to the dataset file(s)"
    )
    parser.add_argument(
        "--task-name", 
        type=str, 
        required=True,
        help="Task name for evaluation (jsquad, wiki_ner, jmmlu, etc.)"
    )
    parser.add_argument(
        "--chat-template", 
        type=str, 
        help="Path to the chat template file"
    )
    parser.add_argument(
        "--few-shots", 
        type=int, 
        default=0,
        help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--subset", 
        type=str,
        choices=["test", "dev"],
        default="test",
        help="Dataset subset (test or dev)"
    )
    parser.add_argument(
        "--test-mode", 
        action="store_true", 
        help="Test mode (use only 1 sample)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override temperature setting"
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=0,
        help="Interval between inference calls in seconds"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./results",
        help="Output directory for evaluation results"
    )
    return parser.parse_args()


def main() -> None:
    """
    メインの実行関数
    
    LLM Leaderboard形式の評価を実行し、結果を保存・表示します。
    コマンドライン引数を解析し、設定に基づいてデータセットをサンプリングして評価します。
    """
    args = parse_args()
    
    # 出力ディレクトリがない場合は作成
    os.makedirs(args.output, exist_ok=True)
    
    # 設定ファイルを読み込む
    try:
        config = Config.from_json(args.config)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)
    
    # コマンドライン引数で上書き
    if args.verbose:
        config.verbose = args.verbose
    
    if args.temperature is not None:
        config.llm.temperature = args.temperature
    
    # 推論間隔を設定
    config.llm.inference_interval = args.inference_interval
    
    # チャットテンプレートを設定
    chat_template_path = args.chat_template
    
    # データセットファイルの取得
    if not args.dataset:
        print("Error: No dataset specified")
        sys.exit(1)
    
    # 結果処理クラスの初期化
    result_processor = ResultProcessor({
        "output_dir": args.output,
        "save_predictions": True,
        "save_metrics": True,
        "result_file_name": f"eval_results_{args.task_name}.json",
        "prediction_file_name": f"predictions_{args.task_name}.json"
    })
    
    # 評価開始
    start_time = time.time()
    print(f"Starting evaluation for task: {args.task_name}")
    print(f"Configuration: {config.llm.model}, Temperature: {config.llm.temperature}")
    print(f"Few-shots: {args.few_shots}, Subset: {args.subset}, Test mode: {args.test_mode}")
    
    # データセットを読み込む
    try:
        # サンプリングを使ってデータセットを読み込む
        all_datasets = {}
        for path in args.dataset:
            dataset_name = os.path.basename(path).split(".")[0]
            dataset = load_dataset_with_sampling(
                dataset_path=path,
                task_name=args.task_name,
                subset=args.subset,
                test_mode=args.test_mode,
                num_few_shots=args.few_shots
            )
            all_datasets[dataset_name] = dataset
            
            print(f"Loaded dataset: {dataset_name}")
            print(f"Number of samples: {len(dataset.samples)}")
            print(f"Number of few-shots: {len(dataset.few_shots)}")
            print(f"Metrics: {dataset.metrics}")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not all_datasets:
        print("No valid datasets found")
        sys.exit(1)
    
    # 拡張LLMクライアントの初期化
    llm_client = ExtendedLLMClient(config.llm)
    
    # 拡張評価クラスの初期化
    evaluator = ExtendedEvaluator()
    
    all_metrics = {}
    
    # 各データセットに対して評価を実行
    for dataset_name, dataset in all_datasets.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # カテゴリに応じて温度パラメータを調整（MT-bench対応）
        if args.task_name in ["writing", "roleplay", "extraction", "math", "coding", "reasoning", "stem", "humanities"]:
            # MT-benchカテゴリごとの温度設定
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
            temp = temperature_override.get(args.task_name, 0.1)
            if args.temperature is None:  # 引数で上書きされていなければ適用
                config.llm.temperature = temp
                print(f"Using MT-bench temperature for category {args.task_name}: {temp}")
        
        # 予測を生成
        predictions = llm_client.generate_predictions_leaderboard(
            dataset=dataset,
            task_name=args.task_name,
            num_few_shots=args.few_shots,
            chat_template_path=chat_template_path
        )
        
        # 評価
        metrics = evaluator.evaluate(predictions, dataset, args.task_name)
        
        # 結果を保存
        all_metrics[dataset_name] = {"metrics": metrics}
        result_processor.save_metrics(metrics, dataset_name)
        result_processor.save_predictions(predictions, dataset, dataset_name)
    
    # 結果のサマリーを表示
    result_processor.print_summary(all_metrics)
    
    # 評価終了
    end_time = time.time()
    print(f"\nEvaluation completed. Time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
