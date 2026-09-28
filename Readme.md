# Indic BPE Tokenizer: Efficient Tokenization for Hindi

Normalization + regex pre-tokenization + Byte Pair Encoding (BPE), trained on a Hindi web corpus and benchmarked against existing LLM tokenizers.

**References**
- Blog: [Implementing A BPE Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html), Sebastian Raschka
- Video: https://www.youtube.com/watch?v=tOMjTCO0htA
- Figure: [DNA BPE and model architecture](https://www.researchgate.net/figure/DNA-BPE-and-model-architecture-a-The-principle-of-BPE-highlighted-on-an-example-sequence_fig1_382492840)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Architecture](#3-architecture)
4. [Step-by-Step](#4-step-by-step)
5. [Implementation](#5-implementation)
6. [Evaluation](#6-evaluation)
7. [Project Structure](#7-project-structure)
8. [References](#8-references)

---

## 1. Problem Statement

General-purpose LLM tokenizers are optimized mainly for high-resource languages such as English. Hindi and other Indic languages get tokenized inefficiently:

- Longer token sequences for the same input.
- Higher inference latency.
- Higher compute and memory requirements (attention cost and KV cache grow with sequence length).
- Higher inference cost (billing is per token).

**Why Hindi suffers:** Devanagari characters take 3 bytes each in UTF-8. A byte-level BPE trained mostly on English has few Devanagari merges, so a single Hindi character often falls back to 2–3 tokens.

This project builds an efficient multilingual/Indic tokenizer using normalization, regex-based pre-tokenization, and BPE, trained on a Hindi-focused web corpus. It is evaluated against existing tokenizers to show more compact Hindi representations without significantly compromising linguistic coverage.

---

## 2. Objectives

- Train a BPE tokenizer on a Hindi web corpus (FineWeb-Edu-style, quality-filtered).
- Compare against existing tokenizers on:
  - token count
  - sequence-length reduction
  - bytes per token
  - inference efficiency
- Keep full coverage: byte-level base vocabulary (256 bytes), so no unknown tokens and lossless round-trip.

| Baseline tokenizer | Source |
|---|---|
| GPT-2 | `tiktoken` → `gpt2` |
| GPT-4 | `tiktoken` → `cl100k_base` |
| GPT-4o | `tiktoken` → `o200k_base` |

---

## 3. Architecture

### Tokenizer pipeline

```
Hindi web corpus
   │
   ▼
1. Normalization        Unicode NFC, drop zero-width junk, collapse whitespace
   │
   ▼
2. Regex pre-tokenizer  split into words / numbers / punctuation (keeps matras inside words)
   │
   ▼
3. Byte-level BPE       UTF-8 bytes → learned merges → token IDs
   │
   ▼
Token IDs ──► Embedding layer ──► LLM
```

### BPE principle (reference figure)

![DNA BPE and model architecture](https://www.researchgate.net/publication/382492840/figure/fig1/AS:11431281274499724@1724987455021/DNA-BPE-and-model-architecture-a-The-principle-of-BPE-highlighted-on-an-example-sequence.png)

*Figure: (a) BPE principle on an example sequence with the tokenization steps and resulting vocabularies. (b) BERT-style model with 12 transformer blocks that embeds tokens and predicts masked tokens.*

*Credit: Sanabria et al., "DNA language model GROVER learns sequence context in the human genome", Nature Machine Intelligence (2024), via [ResearchGate](https://www.researchgate.net/figure/DNA-BPE-and-model-architecture-a-The-principle-of-BPE-highlighted-on-an-example-sequence_fig1_382492840). Copyright belongs to the original authors/publisher. This figure applies BPE to DNA; the principle is the same for text. If the image does not load, download it to `assets/` and update the path.*

---

## 4. Step-by-Step

### Step 1: Prepare the corpus
- Use a Hindi web corpus, e.g. the Hindi (`hin_Deva`) subset of FineWeb-2. Original FineWeb-Edu is English-only, so verify the subset and license before use.
- Sample a training split and a held-out evaluation split.

### Step 2: Normalize
- Unicode NFC (canonical composed form) so identical text has identical bytes.
- Remove zero-width space and BOM.
- Keep ZWJ/ZWNJ: they change how Devanagari conjuncts render.
- Collapse repeated spaces/tabs.

### Step 3: Regex pre-tokenization
- Split text into chunks before BPE so merges never cross word boundaries.
- Word pattern must include `\p{M}` (combining marks: matras, virama, anusvara, nukta). Using only `\p{L}` splits Hindi words mid-syllable.
- Attach the leading space to the following word.
- Split digits in groups of up to 3.

### Step 4: Convert to bytes
- Encode each chunk as UTF-8 bytes (IDs 0–255 form the base vocabulary).

### Step 5: Train BPE
- Count word frequencies over unique chunks.
- Repeatedly find the most frequent adjacent pair, assign a new ID (starting at 256), replace it everywhere, record the merge.
- Stop at target vocab size (hyperparameter) or when no pair occurs more than once.

### Step 6: Encode
- Normalize → pre-tokenize → bytes → apply learned merges in order (earliest learned first).

### Step 7: Decode
- Map IDs back to bytes (merged IDs expand into their two parts), join, decode as UTF-8.

### Step 8: Evaluate
- Run the same held-out Hindi text through the new tokenizer and each baseline.
- Compute the metrics in [Evaluation](#6-evaluation).

---

## 5. Implementation

Save as `indic_bpe.py`. Requires `regex` (`pip install regex`).

```python
import unicodedata
from collections import Counter

import regex as re

# Keep ZWJ (U+200D) / ZWNJ (U+200C): they change Devanagari conjunct rendering.
_DROP = {"\u200b", "\ufeff"}

# \p{M} (matras, virama, anusvara, nukta) must stay inside words,
# otherwise Hindi words are split mid-syllable.
PAT = re.compile(
    r""" ?[\p{L}\p{M}]+| ?\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"""
)


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c not in _DROP)
    return re.sub(r"[ \t]+", " ", text)


def pretokenize(text):
    return PAT.findall(normalize(text))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(texts, vocab_size):
    freq = Counter()
    for t in texts:
        freq.update(pretokenize(t))
    words = {w: list(w.encode("utf-8")) for w in freq}
    merges = {}
    for new_id in range(256, vocab_size):
        counts = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                counts[p] += freq[w]
        if not counts:
            break
        pair, c = counts.most_common(1)[0]
        if c < 2:
            break
        merges[pair] = new_id
        for w in words:
            words[w] = merge(words[w], pair, new_id)
    return merges


def _encode_chunk(chunk, merges):
    ids = list(chunk.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def encode(text, merges):
    return [i for ch in pretokenize(text) for i in _encode_chunk(ch, merges)]


def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), n in merges.items():
        vocab[n] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def evaluate(encode_fn, texts):
    tokens = sum(len(encode_fn(t)) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    words = sum(len(t.split()) for t in texts)
    return {
        "tokens": tokens,
        "bytes_per_token": n_bytes / tokens,
        "fertility_tokens_per_word": tokens / words,
    }
```

Notes:
- Training recounts all pairs every iteration: fine for learning and small samples, too slow for a full corpus. For scale, use incremental pair-count updates or Hugging Face `tokenizers` with the same normalizer and regex.
- Round-trip is lossless up to normalization: `decode(encode(x)) == normalize(x)`.

### Usage

```python
from indic_bpe import train, encode, decode, evaluate

texts = [...]  # Hindi training documents
merges = train(texts, vocab_size=32000)

ids = encode("भारत में हिंदी शिक्षा का महत्व है।", merges)
print(decode(ids, merges))
```

### Benchmark against baselines

```python
import tiktoken

held_out = [...]  # Hindi evaluation documents

ours = evaluate(lambda t: encode(t, merges), held_out)
baselines = {}
for name in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(name)
    baselines[name] = evaluate(enc.encode, held_out)

for name, m in baselines.items():
    reduction = 1 - ours["tokens"] / m["tokens"]
    print(f"vs {name}: {reduction:.1%} fewer tokens")
```

---

## 6. Evaluation

Same held-out Hindi text for every tokenizer.

| Metric | Definition | Better |
|---|---|---|
| Token count | Total tokens for the eval set | Lower |
| Sequence-length reduction | `1 − tokens_ours / tokens_baseline` | Higher |
| Bytes per token | UTF-8 bytes ÷ tokens | Higher |
| Fertility | Tokens per whitespace-separated word | Lower |
| Round-trip | `decode(encode(x)) == normalize(x)` | 100% |

### Inference efficiency
- Compute proxy: tokens per document (fewer tokens = fewer forward passes and less KV cache).
- Measured: encode throughput (tokens/sec) and end-to-end latency on a fixed prompt set with the same model.
- Cost proxy: `cost ∝ token count`, so the sequence-length reduction maps directly to relative cost.

### Coverage check
- Byte-level base vocabulary: 0 unknown tokens by construction.
- Also test on mixed Hindi/English text and numerals to confirm no regression on English.

### Results

| Tokenizer | Vocab size | Total tokens | Bytes/token | Fertility | Reduction vs baseline |
|---|---|---|---|---|---|
| GPT-2 | 50,257 | TBD | TBD | TBD | n/a |
| GPT-4 (`cl100k_base`) | 100,256 | TBD | TBD | TBD | n/a |
| GPT-4o (`o200k_base`) | 199,997 | TBD | TBD | TBD | n/a |
| **Indic BPE (ours)** | TBD | TBD | TBD | TBD | TBD |

---

## 7. Project Structure

```
.
├── README.md
├── indic_bpe.py        # normalize, pre-tokenize, train, encode, decode, evaluate
├── data/               # train / held-out Hindi splits
├── assets/             # figures
└── results/            # benchmark outputs
```

---

## 8. References

- Raschka, S. (2025). [Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html). Notebook: [LLMs-from-scratch, ch02/05_bpe-from-scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
- Gage, P. (1994). *A New Algorithm for Data Compression*.
- Sanabria, M., Hirsch, J., Joubert, P., Poetsch, A. R. (2024). [DNA language model GROVER learns sequence context in the human genome](https://www.researchgate.net/publication/382492840_DNA_language_model_GROVER_learns_sequence_context_in_the_human_genome). *Nature Machine Intelligence*.
- Video: https://www.youtube.com/watch?v=tOMjTCO0htA
- [tiktoken](https://github.com/openai/tiktoken) · [minbpe](https://github.com/karpathy/minbpe) · [Hugging Face tokenizers](https://github.com/huggingface/tokenizers)
