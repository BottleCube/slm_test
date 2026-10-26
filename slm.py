"""最小規模の SLM をゼロから学習し、GGUF に書き出して Ollama で動かす。

アーキテクチャは Llama 互換（RMSNorm / RoPE / SwiGLU）。トークナイザは文字単位で、語彙は
「学習データに出てきた文字（初出順）」+ 未知文字用のバイトフォールバック。文を追加しても既存の ID は変わらない。

使い方:
  uv run slm.py add "新しい文"   # 文を追加 → 再学習 → Ollama に登録 → 評価プロンプトで応答を記録
  uv run slm.py add-file a.txt   # ファイルを文に分割し、1 文ずつ add と同じ処理（--all で全文まとめて 1 ステップ）
  uv run slm.py train            # 現在のデータで学習して Ollama に登録（記録はしない）
  uv run slm.py replay           # data/sentences.txt を 1 文目から順に増やして全ステップを再実行
  uv run slm.py history          # 評価プロンプトごとに応答の変化を表示
  uv run slm.py generate "プロンプト"  # PyTorch で直接生成（GGUF 書き出しの検証用）
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
SENTENCES = ROOT / "data" / "sentences.txt"
PROMPTS = ROOT / "data" / "prompts.txt"
RUNS = ROOT / "runs"
HISTORY = ROOT / "results" / "history.jsonl"
MODEL_NAME = "slm"
OLLAMA_URL = "http://localhost:11434"

# ---------------------------------------------------------------- tokenizer

UNK, BOS, EOS, PAD = 0, 1, 2, 3
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
BYTE_OFFSET = len(SPECIALS)  # 4..259 = <0x00>..<0xFF>
SPACE = "▁"  # SentencePiece 流に空白はこの文字で表す


class Tokenizer:
    """文字単位 + バイトフォールバック（SentencePiece 形式で GGUF に書ける）。

    学習データに無い文字は UTF-8 バイト列に分解される。バイトトークンは学習の正解に
    一度も現れないので、モデルが出力することは実質無い（＝Ollama に不正な UTF-8 が渡らない）。
    """

    def __init__(self, sentences: list[str]):
        chars = dict.fromkeys(c for s in sentences for c in s.replace(" ", SPACE))
        self.pieces = SPECIALS + [f"<0x{b:02X}>" for b in range(256)] + list(chars)
        self.ids = {p: i for i, p in enumerate(self.pieces)}

    def __len__(self):
        return len(self.pieces)

    def encode(self, text: str) -> list[int]:
        out = []
        for c in text.replace(" ", SPACE):
            if c in self.ids:
                out.append(self.ids[c])
            else:
                out.extend(BYTE_OFFSET + b for b in c.encode("utf-8"))
        return out

    def decode(self, ids: list[int]) -> str:
        buf, out = bytearray(), []
        for i in ids:
            if BYTE_OFFSET <= i < BYTE_OFFSET + 256:
                buf.append(i - BYTE_OFFSET)
            elif i >= BYTE_OFFSET + 256:
                buf.extend(self.pieces[i].encode("utf-8"))
        return buf.decode("utf-8", errors="replace").replace(SPACE, " ")


# ---------------------------------------------------------------- model


@dataclass
class Config:
    vocab_size: int
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 2
    ffn_dim: int = 256
    ctx: int = 128
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, cos, sin):
    # ggml の LLAMA 系 RoPE（mode 0）と同じく、隣り合う 2 要素をペアにして回転させる
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.wq = nn.Linear(c.dim, c.dim, bias=False)
        self.wk = nn.Linear(c.dim, c.dim, bias=False)
        self.wv = nn.Linear(c.dim, c.dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.w_gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.attn_norm(x)
        q, k, v = (w(h).view(B, T, self.c.n_heads, self.c.head_dim).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ffn_norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class SLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.output = nn.Linear(c.dim, c.vocab_size, bias=False)
        inv_freq = c.rope_base ** (-torch.arange(0, c.head_dim, 2).float() / c.head_dim)
        angles = torch.outer(torch.arange(c.ctx).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, tok: Tokenizer, prompt: str, max_new: int = 64) -> str:
        ids = [BOS] + tok.encode(prompt)
        out = []
        for _ in range(max_new):
            nxt = int(self(torch.tensor([ids[-self.c.ctx :]]))[0, -1].argmax())
            if nxt == EOS:
                break
            ids.append(nxt)
            out.append(nxt)
        return tok.decode(out)


# ---------------------------------------------------------------- training


def load_sentences() -> list[str]:
    if not SENTENCES.exists():
        return []
    return [s.strip() for s in SENTENCES.read_text().splitlines() if s.strip() and not s.startswith("#")]


def split_sentences(text: str) -> list[str]:
    """行ごと、さらに 。！？!? や「. 」の後ろで区切る（直後が閉じ括弧なら区切らない）。"""
    out = []
    for line in text.splitlines():
        for s in re.split(r"(?<=[。！？!?])(?![」』）)])|(?<=\.)\s+", line):
            s = s.strip()
            if s and not s.startswith("#"):  # sentences.txt では # 行はコメント扱いになるため除外
                out.append(s)
    return out


def append_sentences(new: list[str]) -> None:
    SENTENCES.parent.mkdir(exist_ok=True)
    with SENTENCES.open("a") as f:
        f.writelines(s + "\n" for s in new)


def train(sentences: list[str], tok: Tokenizer, c: Config, steps: int, lr: float, batch: int, seed: int) -> tuple[SLM, float]:
    torch.manual_seed(seed)
    model = SLM(c)
    seqs = [torch.tensor(([BOS] + tok.encode(s) + [EOS])[: c.ctx + 1]) for s in sentences]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        for g in opt.param_groups:  # cosine decay
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(len(seqs), (min(batch, len(seqs)),))
        x = nn.utils.rnn.pad_sequence([seqs[i] for i in idx], batch_first=True, padding_value=PAD)
        logits = model(x[:, :-1])
        target = x[:, 1:].masked_fill(x[:, 1:] == PAD, -100)
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), target.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    model.eval()
    return model, loss.item()


# ---------------------------------------------------------------- export


def export_gguf(model: SLM, tok: Tokenizer, path: Path) -> None:
    c = model.c
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("slm")
    w.add_context_length(c.ctx)
    w.add_embedding_length(c.dim)
    w.add_block_count(c.n_layers)
    w.add_feed_forward_length(c.ffn_dim)
    w.add_head_count(c.n_heads)
    w.add_head_count_kv(c.n_heads)
    w.add_rope_dimension_count(c.head_dim)
    w.add_rope_freq_base(c.rope_base)
    w.add_layer_norm_rms_eps(c.norm_eps)
    w.add_vocab_size(c.vocab_size)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    types = [gguf.TokenType.UNKNOWN] + [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.BYTE] * 256
    types += [gguf.TokenType.NORMAL] * (len(tok) - len(types))
    w.add_tokenizer_model("llama")
    w.add_token_list(tok.pieces)
    w.add_token_scores([0.0] * len(tok))  # 複数文字のピースが無いので結合は起きない
    w.add_token_types(types)
    w.add_unk_token_id(UNK)
    w.add_bos_token_id(BOS)
    w.add_eos_token_id(EOS)
    w.add_pad_token_id(PAD)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    w.add_add_space_prefix(False)

    def t(x):
        return x.detach().float().numpy()

    w.add_tensor("token_embd.weight", t(model.embed.weight))
    for i, b in enumerate(model.blocks):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", t(b.attn_norm.weight))
        w.add_tensor(p + "attn_q.weight", t(b.wq.weight))
        w.add_tensor(p + "attn_k.weight", t(b.wk.weight))
        w.add_tensor(p + "attn_v.weight", t(b.wv.weight))
        w.add_tensor(p + "attn_output.weight", t(b.wo.weight))
        w.add_tensor(p + "ffn_norm.weight", t(b.ffn_norm.weight))
        w.add_tensor(p + "ffn_gate.weight", t(b.w_gate.weight))
        w.add_tensor(p + "ffn_up.weight", t(b.w_up.weight))
        w.add_tensor(p + "ffn_down.weight", t(b.w_down.weight))
    w.add_tensor("output_norm.weight", t(model.norm.weight))
    w.add_tensor("output.weight", t(model.output.weight))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


MODELFILE = """FROM ./model.gguf
TEMPLATE \"\"\"{{ .Prompt }}\"\"\"
PARAMETER temperature 0
PARAMETER seed 0
PARAMETER num_ctx {ctx}
PARAMETER num_predict 64
"""


def ollama_create(run_dir: Path, tag: str, ctx: int) -> None:
    (run_dir / "Modelfile").write_text(MODELFILE.replace("{ctx}", str(ctx)))
    subprocess.run(["ollama", "create", f"{MODEL_NAME}:{tag}", "-f", "Modelfile"], cwd=run_dir, check=True, capture_output=True)
    subprocess.run(["ollama", "cp", f"{MODEL_NAME}:{tag}", f"{MODEL_NAME}:latest"], check=True, capture_output=True)


def ollama_generate(tag: str, prompt: str) -> str:
    body = json.dumps({"model": f"{MODEL_NAME}:{tag}", "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["response"]


# ---------------------------------------------------------------- experiment


def run_step(sentences: list[str], args, record: bool, n_added: int = 1) -> dict:
    n = len(sentences)
    added = " / ".join(sentences[-n_added:])
    tok = Tokenizer(sentences)
    c = Config(vocab_size=len(tok), dim=args.dim, n_layers=args.layers, n_heads=args.heads, ffn_dim=args.ffn, ctx=args.ctx)
    print(f"[{n} 文で学習] 追加: {added}")
    model, loss = train(sentences, tok, c, args.steps, args.lr, args.batch, args.seed)
    n_params = sum(p.numel() for p in model.parameters())

    tag = f"s{n:03d}"
    run_dir = RUNS / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(c), "pieces": tok.pieces, "state_dict": model.state_dict()}, run_dir / "model.pt")
    export_gguf(model, tok, run_dir / "model.gguf")
    ollama_create(run_dir, tag, c.ctx)
    print(f"  → ollama run {MODEL_NAME}:{tag}  (params {n_params:,}, vocab {len(tok)}, loss {loss:.4f})")

    prompts = [p for p in PROMPTS.read_text().splitlines() if p.strip()] if PROMPTS.exists() else []
    outputs = {p: ollama_generate(tag, p) for p in prompts}
    for p, o in outputs.items():
        print(f"  {p!r} → {o!r}")
    entry = {"step": n, "sentence": added, "loss": round(loss, 4), "params": n_params, "outputs": outputs}
    if record:
        HISTORY.parent.mkdir(exist_ok=True)
        with HISTORY.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def cmd_add(args):
    append_sentences([args.sentence.strip()])
    run_step(load_sentences(), args, record=True)


def cmd_add_file(args):
    new = split_sentences(Path(args.path).read_text())
    if not new:
        sys.exit(f"{args.path} に文がありません")
    print(f"{args.path}: {len(new)} 文")
    if args.all:
        append_sentences(new)
        run_step(load_sentences(), args, record=True, n_added=len(new))
        return
    for s in new:
        append_sentences([s])
        run_step(load_sentences(), args, record=True)


def cmd_train(args):
    sentences = load_sentences()
    if not sentences:
        sys.exit(f"{SENTENCES} が空です")
    run_step(sentences, args, record=False)


def cmd_replay(args):
    sentences = load_sentences()
    if not sentences:
        sys.exit(f"{SENTENCES} が空です")
    HISTORY.unlink(missing_ok=True)
    shutil.rmtree(RUNS, ignore_errors=True)
    for k in range(1, len(sentences) + 1):
        run_step(sentences[:k], args, record=True)


def cmd_history(args):
    if not HISTORY.exists():
        sys.exit("まだ記録がありません。`uv run slm.py add \"文\"` から始めてください。")
    entries = [json.loads(line) for line in HISTORY.read_text().splitlines()]
    print("== 追加した文 ==")
    for e in entries:
        print(f"  s{e['step']:03d}  loss {e['loss']:.4f}  {e['sentence']}")
    for p in dict.fromkeys(p for e in entries for p in e["outputs"]):
        print(f"\n== プロンプト: {p!r} ==")
        prev = None
        for e in entries:
            o = e["outputs"].get(p)
            mark = " " if o == prev else "*"  # * = 前ステップから応答が変わった
            print(f" {mark}s{e['step']:03d}  {o!r}")
            prev = o


def cmd_generate(args):
    runs = sorted(RUNS.glob("s*/model.pt"))
    path = RUNS / args.tag / "model.pt" if args.tag else (runs[-1] if runs else None)
    if not path or not path.exists():
        sys.exit("学習済みモデルがありません")
    ckpt = torch.load(path)
    tok = Tokenizer([])
    tok.pieces = ckpt["pieces"]
    tok.ids = {p: i for i, p in enumerate(tok.pieces)}
    model = SLM(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(model.generate(tok, args.prompt))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def train_opts(p):
        p.add_argument("--dim", type=int, default=64)
        p.add_argument("--layers", type=int, default=2)
        p.add_argument("--heads", type=int, default=2)
        p.add_argument("--ffn", type=int, default=256)
        p.add_argument("--ctx", type=int, default=128)
        p.add_argument("--steps", type=int, default=500)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("add", help="文を 1 つ追加して再学習・評価")
    p.add_argument("sentence")
    train_opts(p)
    p.set_defaults(func=cmd_add)
    p = sub.add_parser("add-file", help="ファイルの文を 1 文ずつ追加して再学習・評価")
    p.add_argument("path")
    p.add_argument("--all", action="store_true", help="1 文ずつではなく全文まとめて 1 ステップで学習")
    train_opts(p)
    p.set_defaults(func=cmd_add_file)
    p = sub.add_parser("train", help="現在のデータで学習して Ollama に登録")
    train_opts(p)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("replay", help="1 文目から順に全ステップを再実行")
    train_opts(p)
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("history", help="応答の変化を表示")
    p.set_defaults(func=cmd_history)
    p = sub.add_parser("generate", help="PyTorch で直接生成")
    p.add_argument("prompt")
    p.add_argument("--tag", help="例: s003（省略時は最新）")
    p.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
