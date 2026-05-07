"""Evaluate a trained tokenizer's compression ratio per language vs cl100k baseline.

Higher bytes/token = better compression = fewer tokens for the same text.
For balanced FR/EN/code training, expect FR compression to improve markedly
over cl100k_base (which is EN-biased), with EN staying comparable.
"""
import argparse
import json

from tokenizers import Tokenizer

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def load_samples(path, n):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                out.append(json.loads(line)["text"])
            except (KeyError, json.JSONDecodeError):
                continue
    return out


def encode_ours(tok, samples):
    total_bytes = sum(len(t.encode("utf-8")) for t in samples)
    total_tokens = sum(len(tok.encode(t).ids) for t in samples)
    return total_bytes, total_tokens


def encode_baseline(enc, samples):
    total_bytes = sum(len(t.encode("utf-8")) for t in samples)
    total_tokens = sum(len(enc.encode(t, disallowed_special=())) for t in samples)
    return total_bytes, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval", nargs="+", required=True,
                    help="name=path pairs, e.g. en=data/en.jsonl fr=data/fr.jsonl")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--baseline", default="o200k_base",
                    help="tiktoken encoding to compare against (e.g. cl100k_base, o200k_base)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    enc_baseline = None
    if HAS_TIKTOKEN:
        try:
            enc_baseline = tiktoken.get_encoding(args.baseline)
        except Exception as e:
            print(f"[warn] could not load tiktoken {args.baseline}: {e}")

    header = f"{'source':<10} {'bytes/tok':>12} {'tokens':>14}"
    if enc_baseline:
        header += f" {'baseline b/t':>14} {'gain':>8}"
    print(header)
    print("-" * len(header))

    for spec in args.eval:
        name, path = spec.split("=", 1)
        samples = load_samples(path, args.n_samples)
        b, t = encode_ours(tok, samples)
        bpt = b / t if t else 0
        line = f"{name:<10} {bpt:>12.3f} {t:>14,}"
        if enc_baseline:
            b2, t2 = encode_baseline(enc_baseline, samples)
            bpt2 = b2 / t2 if t2 else 0
            gain = (bpt / bpt2 - 1) * 100 if bpt2 else 0
            line += f" {bpt2:>14.3f} {gain:>+7.1f}%"
        print(line)


if __name__ == "__main__":
    main()
