# slm — Ollama で動く最小規模の言語モデル

PyTorch で Llama 互換の小さなモデルをゼロから学習し、GGUF に書き出して Ollama に登録する。

## 目的

1. 最小規模のモデルを作ってみる（デフォルト約 17 万パラメータ、F32 で約 0.7MB）
2. 1 文ずつ学習データを追加して、応答がどう変わるかを観察する

## 準備

```sh
uv sync
ollama serve   # 別ターミナル（Ollama.app が起動していれば不要）
```

## 使い方

```sh
uv run slm.py add "猫はかわいい。"     # 文を追加 → 学習 → ollama に slm:s001 として登録 → 評価
uv run slm.py add "今日は晴れです。"   # → slm:s002
uv run slm.py history                  # 評価プロンプトごとの応答の変化（* = 前回から変化）
ollama run slm:s002 "猫は"             # 任意のステップのモデルと直接対話（slm:latest は最新）
```

| コマンド | 内容 |
|---|---|
| `add "文"` | `data/sentences.txt` に追記し、全文で学習し直して記録する |
| `add-file a.txt` | ファイルを文に分割し、1 文ずつ `add` と同じ処理を繰り返す（`--all` で全文まとめて 1 ステップ） |
| `train` | 現在のデータで学習・登録だけ行う（記録しない） |
| `replay` | 1 文目から順に増やして全ステップを再実行（設定を変えて比較したいとき） |
| `history` | `results/history.jsonl` の内容を見やすく表示 |
| `generate "プロンプト"` | Ollama を通さず PyTorch で生成（変換の検証用） |

`add-file` の文分割: 改行で区切り、さらに `。！？!?` の後ろ（直後が `」』）)` なら区切らない）と、英文の「`. `」で区切る。`#` で始まる文は除外。

評価プロンプトは `data/prompts.txt`（1 行 1 プロンプト）。
モデルサイズや学習ステップは `--dim --layers --heads --ffn --ctx --steps --lr --seed` で変更できる。

## 設計メモ

- **アーキテクチャ**: Llama（RMSNorm / RoPE / SwiGLU / bias なし）。llama.cpp・Ollama がそのまま読める
  `general.architecture = llama` の GGUF を `gguf` パッケージで直接書き出している（llama.cpp のクローンは不要）。
- **トークナイザ**: 文字単位。語彙は「特殊トークン 4 + バイトフォールバック 256 + 学習データの文字（初出順）」。
  文を追加しても既存 ID は変わらない。未知の文字はバイトに分解されるが、バイトは学習の正解に現れないため出力されない。
  （純粋なバイト単位だと、学習不足のモデルが 文字の途中のバイトを出力して文字化けするため、この方式にした。）
- **毎回ゼロから学習**: 各ステップは同じシード・同じステップ数で最初から学習し直す。違いが「データが 1 文増えたこと」だけになるようにするため。
- **プロンプト形式**: チャットテンプレートは無く、プロンプトの続きを生成するだけ（`TEMPLATE {{ .Prompt }}`、temperature 0）。
- **成果物**: `runs/sNNN/` に `model.pt`・`model.gguf`・`Modelfile`、`results/history.jsonl` に各ステップの応答を保存する。

## ライセンス

[0BSD](LICENSE)
