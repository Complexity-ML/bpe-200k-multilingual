"""Stream-sample FR/EN/code corpora from HuggingFace into JSONL files.

Defaults match the recommended mix for a ~200k vocab balanced FR/EN/code BPE.
Streaming is used so we never download the full datasets (FineWeb is multi-TB,
starcoderdata is ~250 GB).

starcoderdata is gated — run `huggingface-cli login` first.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

# Curated language mix for code. Equal weighting by default → starcoderdata
# is already volume-balanced, interleave gives ~uniform per-lang sampling.
DEFAULT_CODE_LANGS = [
    "python", "rust", "javascript", "typescript", "go",
    "java", "cpp", "c", "ruby", "shell", "sql", "html",
]

SOURCES = {
    "en": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_field": "text",
    },
    "fr": {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "fra_Latn",
        "split": "train",
        "text_field": "text",
    },
    "code": {
        "path": "bigcode/starcoderdata",
        "split": "train",
        "text_field": "content",
        # filled at runtime from --code-langs
        "langs": None,
    },
}


def build_stream(key, seed, shuffle_buffer):
    cfg = SOURCES[key]
    if key == "code":
        # One stream per language sub-dir, then interleave with equal probs.
        # select_columns drops metadata that has inconsistent types across
        # language sub-dirs (e.g. max_stars_count int64 vs float64), which
        # would otherwise break interleave_datasets feature-alignment.
        streams = [
            load_dataset(
                cfg["path"],
                data_dir=lang,
                split=cfg["split"],
                streaming=True,
            ).select_columns([cfg["text_field"]]).shuffle(seed=seed, buffer_size=shuffle_buffer)
            for lang in cfg["langs"]
        ]
        return interleave_datasets(streams, seed=seed, stopping_strategy="all_exhausted")
    return load_dataset(
        cfg["path"],
        name=cfg.get("name"),
        split=cfg["split"],
        streaming=True,
        trust_remote_code=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)


def sample_source(key, target_bytes, out_path, seed=42, shuffle_buffer=10_000):
    cfg = SOURCES[key]
    ds = build_stream(key, seed, shuffle_buffer)

    written = 0
    docs = 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc=key)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(cfg["text_field"])
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n = len(text.encode("utf-8"))
            written += n
            docs += 1
            pbar.update(n)
            if written >= target_bytes:
                break
    pbar.close()
    return written, docs


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    if s.endswith("GB"): mult, s = 1024**3, s[:-2]
    elif s.endswith("MB"): mult, s = 1024**2, s[:-2]
    elif s.endswith("KB"): mult, s = 1024, s[:-2]
    elif s.endswith("B"): s = s[:-1]
    return int(float(s) * mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--en", default="15GB", help="bytes of EN to sample (e.g. 15GB)")
    ap.add_argument("--fr", default="15GB")
    ap.add_argument("--code", default="12GB")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", nargs="+", default=["en", "fr", "code"])
    ap.add_argument("--code-langs", nargs="+", default=DEFAULT_CODE_LANGS,
                    help="starcoderdata language sub-dirs to interleave")
    args = ap.parse_args()

    SOURCES["code"]["langs"] = args.code_langs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {"en": parse_size(args.en), "fr": parse_size(args.fr), "code": parse_size(args.code)}

    for key in args.sources:
        out = out_dir / f"{key}.jsonl"
        written, docs = sample_source(key, targets[key], out, seed=args.seed)
        print(f"{key}: {written / 1024**3:.2f} GB / {docs:,} docs -> {out}")


if __name__ == "__main__":
    main()
