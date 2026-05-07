"""Train a byte-level BPE tokenizer with the o200k_base pre-tokenization regex.

Reproduces the tiktoken recipe used by GPT-4o / GPT-5:
  - o200k regex pre-split (smarter Unicode handling, multilingual-friendly)
  - byte-level fallback (256 base tokens guarantee zero UNK on any UTF-8 input)
  - GPT-style special tokens incl. FIM markers for code
"""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence as PreSequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tqdm import tqdm

from sample_corpus import SOURCES, build_stream, parse_size, DEFAULT_CODE_LANGS

# o200k_base pre-tokenization pattern (the regex used by GPT-4o / GPT-5).
# Better than cl100k for non-English text: caps word length, splits long
# digit runs into 1-3 char chunks, and is friendlier to CJK / accented FR.
# tokenizers.Regex uses fancy-regex, which supports the (?!\S) lookahead.
O200K_PAT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
]


def iter_texts(jsonl_paths):
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)["text"]
                except (KeyError, json.JSONDecodeError):
                    continue


def iter_streaming(en_bytes, fr_bytes, code_bytes, code_langs, seed):
    """Yield texts straight from HF streaming, capped per source by byte budget."""
    SOURCES["code"]["langs"] = code_langs
    budgets = [("en", en_bytes), ("fr", fr_bytes), ("code", code_bytes)]
    for key, target in budgets:
        if target <= 0:
            continue
        cfg = SOURCES[key]
        ds = build_stream(key, seed=seed, shuffle_buffer=10_000)
        written = 0
        pbar = tqdm(total=target, unit="B", unit_scale=True, desc=key)
        for row in ds:
            text = row.get(cfg["text_field"])
            if not text:
                continue
            yield text
            n = len(text.encode("utf-8"))
            written += n
            pbar.update(n)
            if written >= target:
                break
        pbar.close()


def build_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = PreSequence([
        Split(pattern=Regex(O200K_PAT), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="JSONL files with .text field")
    ap.add_argument("--streaming", action="store_true",
                    help="Train directly from HF streaming, no JSONL spill (needs HF auth)")
    ap.add_argument("--en", default="15GB", help="streaming: bytes of EN to consume")
    ap.add_argument("--fr", default="15GB")
    ap.add_argument("--code", default="12GB")
    ap.add_argument("--code-langs", nargs="+", default=DEFAULT_CODE_LANGS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=200_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    if not args.streaming and not args.data:
        ap.error("provide --data <jsonl files> or --streaming")

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # 256-byte base alphabet -> any UTF-8 sequence is encodable
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    if args.streaming:
        iterator = iter_streaming(
            parse_size(args.en), parse_size(args.fr), parse_size(args.code),
            args.code_langs, args.seed,
        )
    else:
        iterator = iter_texts(args.data)

    tokenizer.train_from_iterator(iterator, trainer=trainer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"Saved {out} (vocab size: {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
