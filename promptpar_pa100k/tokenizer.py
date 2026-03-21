from __future__ import annotations

import gzip
import html
from functools import lru_cache

import ftfy
import regex as re
import torch

from .config import DEFAULT_CONFIG


def default_bpe_path() -> str:
    path = DEFAULT_CONFIG.tokenizer_path
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer asset not found at '{path}'.")
    return str(path)


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for value in range(2**8):
        if value not in bs:
            bs.append(value)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(value) for value in cs]))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SimpleTokenizer:
    def __init__(self, bpe_path: str | None = None) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        merges = gzip.open(bpe_path or default_bpe_path()).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [item + "</w>" for item in vocab]
        for merge in merge_pairs:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break

                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        clean_text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, clean_text):
            encoded = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(encoded).split(" "))
        return bpe_tokens


_TOKENIZER = SimpleTokenizer()


def tokenize(texts: str | list[str], context_length: int = DEFAULT_CONFIG.context_length, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _TOKENIZER.encoder["<|startoftext|>"]
    eot_token = _TOKENIZER.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)

    for index, text in enumerate(texts):
        tokens = [sot_token] + _TOKENIZER.encode(text) + [eot_token]
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f"Input '{text}' is too long for context length {context_length}.")
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    return result
