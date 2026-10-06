"""One-time conversion of the released tokenizer to a portable tiktoken artifact."""

import argparse
import base64
import hashlib
import json
import random
import tiktoken
from pathlib import Path
from .tokenizer import LFMTokenizer


def byte_decoder():
    original = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    remaining = [b for b in range(256) if b not in original]
    return dict(
        zip(
            map(chr, original + list(range(256, 256 + len(remaining)))),
            original + remaining,
        )
    )


def validate_conversion(raw, tokenizer):
    """Conversion-only reference check; tokenizers is not a runtime dependency."""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(
            "Conversion validation needs tokenizers: run with uv run --with tokenizers"
        ) from error
    reference = Tokenizer.from_str(json.dumps(raw))
    decoder = byte_decoder()
    literals = {item["content"] for item in raw["added_tokens"]}
    words = [
        bytes(decoder[c] for c in token).decode("utf8", errors="replace")
        for token in raw["model"]["vocab"]
        if token not in literals
    ]
    samples = words + list(literals)
    samples += ["\n" * n + " " * n + "\t" * n for n in range(1, 150)]
    samples += [
        "python apythonic Mathias",
        "你好 مرحبا café e\u0301 🐍",
        "",
        "<image><|im_end|>",
    ]
    rng = random.Random(17)
    samples += ["".join(rng.choices(words, k=5)) for _ in range(10000)]
    for text in samples:
        expected = reference.encode(text, add_special_tokens=False).ids
        actual = tokenizer.encode(text, add_special_tokens=False)
        if actual != expected:
            raise ValueError(f"Tiktoken conversion changes tokenization for {text!r}")
        if tokenizer.decode(
            actual, clean_up_tokenization_spaces=False
        ) != reference.decode(expected, skip_special_tokens=False):
            raise ValueError(f"Tiktoken conversion changes decoding for {text!r}")
    return len(samples)


def convert_tokenizer(source, destination):
    source, destination = Path(source), Path(destination)
    raw_bytes = (source / "tokenizer.json").read_bytes()
    raw = json.loads(raw_bytes)
    model = raw["model"]
    if (
        model["type"] != "BPE"
        or raw["normalizer"] is not None
        or model.get("dropout") is not None
    ):
        raise ValueError("Expected deterministic byte-level BPE without normalization")
    for flag in ("byte_fallback", "ignore_merges"):
        if model.get(flag):
            raise ValueError(f"Unsupported BPE option: {flag}")
    pre = raw["pre_tokenizer"]["pretokenizers"]
    if (
        len(pre) != 2
        or pre[0]["type"] != "Split"
        or pre[1]["type"] != "ByteLevel"
        or pre[1]["use_regex"]
        or pre[1]["add_prefix_space"]
    ):
        raise ValueError("Unsupported pre-tokenizer structure")
    added = raw["added_tokens"]
    if any(a["single_word"] or a["lstrip"] or a["rstrip"] for a in added):
        raise ValueError(
            "Added-token boundary/whitespace rules require a different adapter"
        )
    decoder = byte_decoder()
    literals = {a["content"]: a["id"] for a in added}
    # HF stores pair priorities separately from output IDs. Tiktoken uses one
    # integer for both, so allocate internal ranks and persist the ID translation.
    ranks = {bytes([b]): b for b in range(256)}
    vocab = model["vocab"]
    rank_to_id = [
        vocab[next(c for c, byte in decoder.items() if byte == b)] for b in range(256)
    ]
    for left, right in model["merges"]:
        token = left + right
        data = bytes(decoder[c] for c in token)
        if token not in literals and data not in ranks:
            ranks[data] = len(ranks)
            rank_to_id.append(vocab[token])
    for token, index in vocab.items():
        data = bytes(decoder[c] for c in token)
        if token not in literals and data not in ranks:
            ranks[data] = len(ranks)
            rank_to_id.append(index)
    literal_ranks = {}
    for token, index in literals.items():
        literal_ranks[token] = len(rank_to_id)
        rank_to_id.append(index)
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    config = dict(
        format_version=1,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        pattern=pre[0]["pattern"]["Regex"],
        rank_to_id=rank_to_id,
        literal_ranks=literal_ranks,
        special_tokens={a["content"]: a["id"] for a in added if a["special"]},
        added_tokens={a["content"]: a["id"] for a in added if not a["special"]},
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=0,
        vocab_size=len(model["vocab"]),
        clean_up_tokenization_spaces=tokenizer_config.get(
            "clean_up_tokenization_spaces", True
        ),
    )
    encoding = tiktoken.Encoding(
        name="lfm2.5-conversion",
        pat_str=config["pattern"],
        mergeable_ranks=ranks,
        special_tokens=literal_ranks,
    )
    candidate = LFMTokenizer(encoding, config)
    config["validation_cases"] = validate_conversion(raw, candidate)
    # Publish only after validating; never replace an artifact with a failed conversion.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "vocab.tiktoken").write_bytes(
        b"".join(
            base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
            for token, rank in sorted(ranks.items(), key=lambda item: item[1])
        )
    )
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    return LFMTokenizer.from_pretrained(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    tok = convert_tokenizer(args.source, args.destination)
    print(
        f"Saved {tok.config['vocab_size']} tokens to {args.destination}; validated {tok.config['validation_cases']} encoding/decoding cases."
    )


if __name__ == "__main__":
    main()
