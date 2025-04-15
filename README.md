# LLM日本語評価スクリプト

このリポジトリは、LLM（大規模言語モデル）の日本語テキスト生成能力を評価するためのツールセットです。特にlitellmを使用して推論を行い、さまざまな評価指標に基づいてモデルのパフォーマンスを評価します。LLM Leaderboard（Nejumi Leaderboard 3）で使用されているサンプリング方法にも対応しています。

## 特徴

- **litellmの活用**: OpenAI API互換の推論サーバーと連携して、一貫したインターフェースでテキスト生成を評価
- **複数の評価指標**: exact_match、char_f1、entity_labeling_acc、pearsonやspearmanなど複数の評価指標をサポート
- **並列処理**: 非同期処理による並列評価をサポート
- **柔軟な設定**: JSON設定ファイルを通じて評価プロセスを細かく制御可能
- **結果の保存と表示**: 評価結果と生成テキストの保存、結果サマリーの表示機能
- **LLM Leaderboardサンプリング**: Nejumi Leaderboard 3で使用されているサンプリング方法を再現し、データセットサイズを適切に調整

## インストール方法

1. 依存ライブラリをインストール:
```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な評価

1. 設定ファイルを準備（`config.json`）:
```json
{
  "llm": {
    "endpoint": "http://localhost:8000/v1/chat/completions",
    "model": "smollm-tiny",
    "temperature": 0.0,
    "max_tokens": 500
  },
  "metrics": {
    "enabled": ["exact_match", "char_f1"]
  },
  "output": {
    "output_dir": "./results"
  },
  "verbose": true
}
```

2. データセットファイルを準備（`dataset.json`）:
```json
{
  "instruction": "質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。",
  "output_length": 15,
  "metrics": ["exact_match", "char_f1"],
  "few_shots": [
    {
      "input": "質問：日本の首都は？",
      "output": "東京"
    }
  ],
  "samples": [
    {
      "input": "質問：日本の四季は何がありますか？",
      "output": "春夏秋冬"
    }
  ]
}
```

3. 評価を実行:
```bash
python evaluate.py --config config.json --dataset dataset.json
```

### 非同期評価の実行

より高速な評価のために非同期モードを使用する:
```bash
python evaluate.py --config config.json --dataset dataset.json --async
```

### 複数データセットの評価

複数のデータセットを一度に評価:
```bash
python evaluate.py --config config.json --dataset dataset1.json dataset2.json
```

またはディレクトリ内のすべてのJSONデータセットを評価:
```bash
python evaluate.py --config config.json --dataset-dir ./datasets
```

### LLM Leaderboardサンプリングの使用

LLM Leaderboardで使用されているサンプリング方法を利用する:
```bash
python evaluate.py --config config.json --dataset dataset.json --sampling --task-name jsquad
```

詳細なサンプリングオプション:
```bash
# JSQuADのテストセットから100サンプルを使用
python evaluate.py --config config.json --dataset jsquad.json --sampling --task-name jsquad

# wiki_nerのテストセットから20サンプルを使用
python evaluate.py --config config.json --dataset wiki_ner.json --sampling --task-name wiki_ner

# jmmluのテストセットから5サンプル、3-shotで評価
python evaluate.py --config config.json --dataset jmmlu.json --sampling --task-name jmmlu --few-shots 3
```

サンプリング機能の詳細については、[SAMPLING.md](./SAMPLING.md)を参照してください。

### LLM Leaderboard完全再現機能の使用

LLM Leaderboardの評価方法を完全に再現（拡張評価指標などを含む）するには：

```bash
python evaluate_leaderboard.py --config config.json --dataset dataset.json --task-name jsquad
```

詳細は[LEADERBOARD.md](./LEADERBOARD.md)を参照してください。

## 詳細設定

### LLM設定

```json
"llm": {
  "endpoint": "http://localhost:8000/v1/chat/completions",  // LLMサーバーのエンドポイント
  "model": "smollm-tiny",                                  // 使用するモデル名
  "temperature": 0.0,                                      // 生成時の温度パラメータ
  "max_tokens": 500,                                       // 最大トークン数
  "timeout": 60,                                           // タイムアウト秒数
  "max_retries": 3,                                        // 失敗時の最大リトライ回数
  "max_concurrent_requests": 5                             // 並列リクエスト数（非同期モード時）
}
```

### 評価指標設定

```json
"metrics": {
  "enabled": ["exact_match", "char_f1", "entity_labeling_acc", "pearson", "spearman"]
}
```

サポートされている評価指標:
- `exact_match`: 完全一致の精度
- `char_f1`: 文字レベルのF1スコア
- `entity_labeling_acc`: エンティティラベリング精度
- `pearson`: ピアソン相関係数
- `spearman`: スピアマン相関係数

### 出力設定

```json
"output": {
  "output_dir": "./results",                   // 結果の出力ディレクトリ
  "save_predictions": true,                    // 予測結果を保存するか
  "save_metrics": true,                        // 評価結果を保存するか
  "result_file_name": "eval_results.json",     // 評価結果のファイル名
  "prediction_file_name": "predictions.json"   // 予測結果のファイル名
}
```

## データセット形式

データセットは以下のフォーマットのJSONファイルとして提供します:

```json
{
  "instruction": "モデルへの指示",
  "output_length": 数値,
  "metrics": ["使用する評価指標のリスト"],
  "few_shots": [
    {
      "input": "入力テキスト",
      "output": "期待される出力"
    },
    ...
  ],
  "samples": [
    {
      "input": "評価用入力テキスト",
      "output": "正解出力"
    },
    ...
  ]
}
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
