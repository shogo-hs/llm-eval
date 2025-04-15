# LLM Leaderboard評価再現ガイド

このドキュメントでは、llm-evalリポジトリでLLM Leaderboard（Nejumi Leaderboard 3）の評価方法を再現する方法について説明します。

## 概要

LLM Leaderboardの評価方法を完全に再現するために、このリポジトリには以下の追加機能が実装されています：

1. **サンプリング機能**: 各タスクに適したサンプル数を自動的に選択
2. **拡張評価指標**: exact_match_figure, set_f1, BLEU, 制御性評価など
3. **プロンプト処理**: MT-bench互換のプロンプト形式
4. **チャットテンプレート**: 異なるモデル用のテンプレート適用機能
5. **litellm統合**: litellmを使用した推論

## インストール方法

必要なライブラリをインストールします：

```bash
pip install -r requirements.txt
```

## 評価の実行方法

LLM Leaderboard形式で評価を実行するには、`evaluate_leaderboard.py`スクリプトを使用します：

```bash
python evaluate_leaderboard.py --config config.json --dataset dataset.json --task-name jsquad
```

### 引数

- `--config`: 設定ファイルのパス（必須）
- `--dataset`: データセットファイルのパス（複数可）（必須）
- `--task-name`: 評価タスク名（例: jsquad, wiki_ner, jmmlu）（必須）
- `--chat-template`: チャットテンプレートファイルのパス（オプション）
- `--few-shots`: few-shotの数（デフォルト: 0）
- `--subset`: データセットのサブセット（test または dev、デフォルト: test）
- `--test-mode`: テストモード（1サンプルのみ使用）
- `--temperature`: 温度パラメータの上書き
- `--inference-interval`: 推論間隔（秒）
- `--output`: 出力ディレクトリ（デフォルト: ./results）
- `--verbose`: 詳細出力の有効化

### タスク名の例

- **翻訳タスク**: alt-e-to-j, alt-j-to-e, wikicorpus-e-to-j, wikicorpus-j-to-e
- **QAタスク**: jsquad, jcommonsenseqa, jemhopqa, niilc, aio
- **推論タスク**: mawps, mgsm
- **エンティティ抽出**: wiki_ner, wiki_coreference, chabsa
- **意味理解**: jnli, janli, jsem, jsick, jamp
- **文法解析**: jcola-in-domain, jcola-out-of-domain, jblimp, wiki_reading, wiki_pas, wiki_dependency
- **多言語**: jmmlu, mmlu_en
- **MT-bench**: writing, roleplay, extraction, math, coding, reasoning, stem, humanities

## サンプリング方法

LLM Leaderboardのサンプリングルールでは、タスクごとに最適なサンプル数が定義されています：

- テストモード: 各データセット1サンプル
- "wiki"データセット: テスト用20サンプル、検証用5サンプル
- "mmlu"データセット: テスト用5サンプル、検証用1サンプル
- その他のデータセット: テスト用100サンプル、検証用10サンプル

## 評価指標

以下の評価指標が実装されています：

- `exact_match`: 完全一致精度
- `exact_match_figure`: 数値一致精度
- `char_f1`: 文字レベルのF1スコア
- `set_f1`: セットベースのF1スコア
- `pearson`: ピアソン相関係数
- `spearman`: スピアマン相関係数
- `bleu_ja`: 日本語BLEU
- `bleu_en`: 英語BLEU
- 制御性評価: 形式の適合度

## 例

JSQuADの評価（0-shot）：

```bash
python evaluate_leaderboard.py --config config.json --dataset jsquad.json --task-name jsquad
```

JMMLUの評価（2-shot）：

```bash
python evaluate_leaderboard.py --config config.json --dataset jmmlu.json --task-name jmmlu --few-shots 2
```

wiki_nerの評価（カスタムチャットテンプレート使用）：

```bash
python evaluate_leaderboard.py --config config.json --dataset wiki_ner.json --task-name wiki_ner --chat-template templates/my_template.jinja
```

MT-benchのroleplayカテゴリを評価：

```bash
python evaluate_leaderboard.py --config config.json --dataset mt_bench_roleplay.json --task-name roleplay
```

## 結果の確認

評価結果と生成された予測は指定された出力ディレクトリに保存されます：

- `eval_results_{task_name}.json`: 評価結果
- `predictions_{task_name}.json`: 生成された予測

## 注意事項

- 評価結果を完全に再現するには、同じモデル、同じサンプリング方法、同じ評価指標を使用することが重要です。
- MT-benchカテゴリでは、LLM Leaderboardと同様にカテゴリごとに適切な温度設定が自動的に適用されます。
