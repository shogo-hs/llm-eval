#!/usr/bin/env python
"""
LLM評価スクリプト

使用例:
    評価スクリプトの実行:
        python evaluate.py --config config.json --dataset dataset.json
    複数データセットの評価:
        python evaluate.py --config config.json --dataset dataset1.json dataset2.json
"""
import os
import sys
import json
import argparse
import asyncio
from typing import Dict, List, Optional, Any
from glob import glob

from llm_jp_evaluator.data.loader import DataLoader, load_multiple_datasets
from llm_jp_evaluator.inference.llm_client import LLMClient
from llm_jp_evaluator.metrics.evaluator import Evaluator
from llm_jp_evaluator.utils.result_processor import ResultProcessor
from llm_jp_evaluator.utils.config import Config


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on Japanese datasets")
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
        "--dataset-dir", 
        type=str, 
        help="Directory containing dataset files"
    )
    parser.add_argument(
        "--async", 
        action="store_true",
        dest="use_async",
        help="Use asynchronous inference"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()


async def evaluate_async(config: Config, datasets: Dict[str, Any]):
    """非同期評価を実行"""
    llm_client = LLMClient(config.llm)
    evaluator = Evaluator(config.metrics.enabled)
    result_processor = ResultProcessor(config.output)
    
    all_metrics = {}
    
    for dataset_name, dataset in datasets.items():
        if config.verbose:
            print(f"Evaluating dataset: {dataset_name}")
            print(f"Number of samples: {len(dataset.samples)}")
            print(f"Metrics to use: {dataset.metrics}")
        
        # 非同期で予測を生成
        predictions = await llm_client.generate_predictions_async(dataset)
        
        # 評価
        metrics = evaluator.evaluate(predictions, dataset)
        
        # 結果を保存
        all_metrics[dataset_name] = {"metrics": metrics}
        result_processor.save_metrics(metrics, dataset_name)
        result_processor.save_predictions(predictions, dataset, dataset_name)
    
    # 結果のサマリーを表示
    result_processor.print_summary(all_metrics)


def evaluate_sync(config: Config, datasets: Dict[str, Any]):
    """同期評価を実行"""
    llm_client = LLMClient(config.llm)
    evaluator = Evaluator(config.metrics.enabled)
    result_processor = ResultProcessor(config.output)
    
    all_metrics = {}
    
    for dataset_name, dataset in datasets.items():
        if config.verbose:
            print(f"Evaluating dataset: {dataset_name}")
            print(f"Number of samples: {len(dataset.samples)}")
            print(f"Metrics to use: {dataset.metrics}")
        
        # 同期で予測を生成
        predictions = llm_client.generate_predictions(dataset)
        
        # 評価
        metrics = evaluator.evaluate(predictions, dataset)
        
        # 結果を保存
        all_metrics[dataset_name] = {"metrics": metrics}
        result_processor.save_metrics(metrics, dataset_name)
        result_processor.save_predictions(predictions, dataset, dataset_name)
    
    # 結果のサマリーを表示
    result_processor.print_summary(all_metrics)


def main():
    """メインの実行関数"""
    args = parse_args()
    
    # 設定ファイルを読み込む
    try:
        config = Config.from_json(args.config)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)
    
    # コマンドライン引数で上書き
    if args.verbose:
        config.verbose = args.verbose
    
    # データセットファイルを取得
    dataset_paths = []
    if args.dataset:
        dataset_paths.extend(args.dataset)
    if args.dataset_dir:
        dataset_paths.extend(glob(os.path.join(args.dataset_dir, "*.json")))
    if not dataset_paths and config.dataset_path:
        dataset_paths.append(config.dataset_path)
    
    if not dataset_paths:
        print("No dataset specified")
        sys.exit(1)
    
    # データセットを読み込む
    try:
        datasets = load_multiple_datasets(dataset_paths)
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        sys.exit(1)
    
    if not datasets:
        print("No valid datasets found")
        sys.exit(1)
    
    # 評価を実行
    try:
        if args.use_async:
            asyncio.run(evaluate_async(config, datasets))
        else:
            evaluate_sync(config, datasets)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
