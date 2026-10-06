"""Load a previously converted local tiktoken artifact; no HF tokenizer at runtime."""

import base64
import json
import re
from pathlib import Path
import tiktoken


class LFMTokenizer:
    def __init__(self, encoding, config):
        self.encoding = encoding
        self.config = config
        self.bos_token_id = config["bos_token_id"]
        self.eos_token_id = config["eos_token_id"]
        self.pad_token_id = config["pad_token_id"]
        self.special_ids = set(config["special_tokens"].values())
        self.rank_to_id = config["rank_to_id"]
        self.id_to_rank = {
            model_id: rank for rank, model_id in enumerate(self.rank_to_id)
        }
        self.literal_ids = config["special_tokens"] | config["added_tokens"]
        self.literal_pattern = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(self.literal_ids, key=len, reverse=True)
            )
        )

    @classmethod
    def from_pretrained(cls, directory=None):
        directory = (
            Path(directory)
            if directory is not None
            else Path(__file__).parent / "tokenizer"
        )
        config = json.loads((directory / "config.json").read_text())
        if config["format_version"] != 1:
            raise ValueError("Unsupported tokenizer artifact version")
        ranks = {}
        for line in (directory / "vocab.tiktoken").read_bytes().splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        # Literal ranks are needed for decoding, including ordinary added tokens.
        literals = config["literal_ranks"]
        encoding = tiktoken.Encoding(
            name="lfm2.5-vl-450m",
            pat_str=config["pattern"],
            mergeable_ranks=ranks,
            special_tokens=literals,
        )
        return cls(encoding, config)

    def encode(self, text, *, add_special_tokens=True):
        # Match literals once in Python. This avoids tiktoken's comparatively
        # expensive scan across 507 special/added tokens on every ordinary piece.
        # The BPE itself remains entirely in tiktoken's Rust implementation.
        ids = [self.bos_token_id] if add_special_tokens else []
        start = 0
        for match in self.literal_pattern.finditer(text):
            ids.extend(
                self.rank_to_id[i]
                for i in self.encoding.encode_ordinary(text[start : match.start()])
            )
            ids.append(self.literal_ids[match.group()])
            start = match.end()
        ids.extend(
            self.rank_to_id[i] for i in self.encoding.encode_ordinary(text[start:])
        )
        return ids

    def encode_batch(self, texts, *, add_special_tokens=True):
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

    def decode(
        self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=None
    ):
        ids = [
            int(i)
            for i in ids
            if not skip_special_tokens or int(i) not in self.special_ids
        ]
        try:
            ranks = [self.id_to_rank[i] for i in ids]
        except KeyError as error:
            raise ValueError(
                f"Model token ID {error.args[0]} is outside the tokenizer vocabulary"
            ) from error
        text = self.encoding.decode(ranks)
        cleanup = (
            self.config["clean_up_tokenization_spaces"]
            if clean_up_tokenization_spaces is None
            else clean_up_tokenization_spaces
        )
        if cleanup:
            for before, after in [
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
            ]:
                text = text.replace(before, after)
        return text

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(ids, **kwargs) for ids in sequences]
