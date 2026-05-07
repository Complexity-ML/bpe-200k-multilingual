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
    ap.add_argument("--data", nargs="+", required=True, help="JSONL files with .text field")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=200_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # 256-byte base alphabet -> any UTF-8 sequence is encodable
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter_texts(args.data), trainer=trainer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"Saved {out} (vocab size: {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
