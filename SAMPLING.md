# LLM Leaderboardサンプリング機能の使い方

## 概要

このリポジトリにはLLM Leaderboard（Nejumi Leaderboard 3）で使用されているサンプリング方法が実装されています。これにより、大規模な評価データセットから、定義されたルールに基づいて適切なサンプル数を選択できます。

## サンプリングルール

LLM Leaderboardのサンプリングルールは以下の通りです：

- テストモード: 各データセット1サンプル
- "wiki"データセット: テスト用20サンプル、検証用5サンプル
- "mmlu"データセット: テスト用5サンプル、検証用1サンプル 
- その他のデータセット: テスト用100サンプル、検証用10サンプル

## コマンドラインからの使用方法

評価スクリプトに`--sampling`オプションが追加されました。

```bash
python evaluate.py --config config.json --dataset dataset.json --sampling --task-name jsquad
```

オプション:
- `--sampling`: LLM Leaderboardのサンプリングルールを使用する
- `--task-name`: タスク名（例: "jsquad", "wiki_ner", "jmmlu"）（必須）
- `--subset`: "test" または "dev"（デフォルト: "test"）
- `--test-mode`: テストモードを有効にする（1サンプルのみを使用）
- `--few-shots`: few-shotの数を指定する

例:
```bash
# JSQuADのテストセットから100サンプルを使用
python evaluate.py --config config.json --dataset jsquad.json --sampling --task-name jsquad

# wiki_nerのテストセットから20サンプルを使用
python evaluate.py --config config.json --dataset wiki_ner.json --sampling --task-name wiki_ner

# jmmluのテストセットから5サンプル、3-shotで評価
python evaluate.py --config config.json --dataset jmmlu.json --sampling --task-name jmmlu --few-shots 3

# テストモード（1サンプルのみ）で評価
python evaluate.py --config config.json --dataset dataset.json --sampling --task-name jsquad --test-mode
```

## プログラムからの使用方法

サンプリング機能は、以下の方法でプログラムから使用できます：

```python
from llm_jp_evaluator.data.loader import load_dataset_with_sampling

# LLM Leaderboardのルールに基づいてデータセットをサンプリング
dataset = load_dataset_with_sampling(
    dataset_path="path/to/dataset.json",
    task_name="jsquad",  # タスク名
    subset="test",       # "test" または "dev"
    test_mode=False,     # テストモードかどうか
    num_few_shots=2      # few-shotの数
)

# サンプリングされたデータセットを使用
print(f"サンプル数: {len(dataset.samples)}")
print(f"few-shot数: {len(dataset.few_shots)}")
```

より詳細なカスタマイズが必要な場合は、`LLMLeaderboardSampler`クラスを直接使用できます：

```python
from llm_jp_evaluator.data.sampler import LLMLeaderboardSampler
from llm_jp_evaluator.data.loader import DataLoader

# サンプラーの初期化
sampler = LLMLeaderboardSampler(seed=42)

# データセットを読み込む
loader = DataLoader("path/to/dataset.json")
original_dataset = loader.load()

# サンプリングを行い、新しいデータセットを生成
sampled_dataset = sampler.create_dataset_with_sampling(
    original_dataset=original_dataset,
    task_name="jsquad",
    subset="test",
    test_mode=False,
    num_few_shots=2
)
```

## MT-benchのサポート

MT-benchデータセットに対するサンプリングと温度設定もサポートしています：

```python
from llm_jp_evaluator.data.sampler import LLMLeaderboardSampler

sampler = LLMLeaderboardSampler()

# MT-benchからサンプリング
questions = sampler.sample_mt_bench("path/to/mt_bench.jsonl", category="writing")

# カテゴリごとの温度設定を取得
temperature = sampler.get_mt_bench_temperature("writing")  # 0.7が返される
```

これにより、LLM Leaderboardの評価方法を忠実に再現できます。
