"""Fixed-point feature contract for the Phase 2 hazardous-API classifier.

This module **is** the M2.0 specification: the only allowed mapping from Python
source text to the ``uint8`` token vector the emitted graph consumes.  Every
rule is integer-only, uses no allocation and no floating point, so the C
frontend of M2.4 can reproduce it byte for byte.

The contract
------------
1. **Normal form** (bytes, ASCII-only, exactly ``WINDOW_BYTES`` long):

   * ``A-Z`` (``0x41..0x5A``) become lowercase (``+0x20``),
   * control bytes (``< 0x20``) and ``0x7F`` become space (``0x20``),
   * bytes ``>= 0x80`` (non-ASCII UTF-8, or the tail of a truncated sequence)
     become ``?`` (``0x3F``); the corpus manifest asserts the training and
     evaluation data is ASCII, so this is a documented policy for unknown
     *bytes* rather than a silent corruption,
   * the first ``WINDOW_BYTES`` bytes are kept and spaces pad the rest.

2. **Token ids** (the graph input: ``TOKEN_COUNT`` uint8 values):

   * ids ``1..96`` are the printable ASCII range: ``byte - 0x1F`` for
     ``0x20 <= byte <= 0x7F``,
   * ids ``97..255`` are hashed bigrams: ``97 + fnv1a32(pair) % 159`` over each
     byte and its successor (with a space sentinel after the last byte, so there
     are exactly ``WINDOW_BYTES`` pairs),
   * the first ``WINDOW_BYTES`` ids are the normal form's bytes and the second
     ``WINDOW_BYTES`` its bigrams, so one flat tensor carries both channels,
   * **id 0 is never emitted.**  Padding happens in the byte domain, so a padded
     position is a real space token; a max-pool bag therefore needs no
     special case for it, and id 0 stays free for a future zero embedding row.

3. **Unknown tokens: there are none.**  The vocabulary is all ``VOCAB_SIZE``
   byte values (the IR's ``EMBEDDING`` addresses indices with ``uint8``), so the
   tokenizer is total; step 1's ``?`` fold is the only lossy rule.

Why these shapes
----------------
The Phase 2 graph is ``EMBEDDING -> MATMUL -> MAXPOOL1D(axis=0) -> ARGMAX``
(``examples/hazard/graph.py``), whose arena is exactly
``TOKEN_COUNT * (dim + labels)``.  With the 2 KB budget that is what chooses the
window: ``TOKEN_COUNT = 96`` costs 1,344 B at ``dim = 8`` / 6 labels, while a
192-token window costs 1,920 B at ``dim = 4`` (see ROADMAP Phase 2).

The tokenizer is otherwise label-agnostic: the label taxonomy is a task decision
(ROADMAP Phase 2, open question 2) and only changes the graph's head.
"""

from __future__ import annotations

from typing import Final, Sequence

__all__ = (
    "BIGRAM_BASE",
    "BIGRAM_BUCKETS",
    "CHARACTER_BASE",
    "CHARACTER_COUNT",
    "TOKEN_COUNT",
    "VOCAB_SIZE",
    "WINDOW_BYTES",
    "detokenize",
    "fnv1a32",
    "normalize_bytes",
    "tokenize",
)

#: Normal-form bytes kept; the snippet is truncated/padded to exactly this many.
WINDOW_BYTES: Final = 48

#: Length of the graph input: byte ids followed by the same number of bigram ids.
TOKEN_COUNT: Final = 2 * WINDOW_BYTES

#: EMBEDDING indices are ``uint8``, so 256 ids are all the IR can address.
VOCAB_SIZE: Final = 256

#: Ids ``1..96``: printable ASCII (``0x20..0x7F``), i.e. ``CHARACTER_COUNT`` of them.
CHARACTER_BASE: Final = 1
CHARACTER_COUNT: Final = 96

#: Ids ``97..255``: one bucket per hashed bigram.
BIGRAM_BASE: Final = CHARACTER_BASE + CHARACTER_COUNT
BIGRAM_BUCKETS: Final = VOCAB_SIZE - BIGRAM_BASE

_SPACE: Final = 0x20
_QUESTION_MARK: Final = 0x3F
#: The single byte value the normal form allows for unknown input.
UNKNOWN_BYTE: Final = _QUESTION_MARK


def fnv1a32(data: bytes) -> int:
    """FNV-1a (32-bit), the hash fastText uses for its subword buckets."""
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def normalize_bytes(source: bytes | str) -> bytes:
    """Returns the fixed-length ASCII normal form of ``source``.

    Text is encoded as UTF-8 first, so a ``str`` and its ``bytes`` form always
    normalise identically.
    """
    raw = source.encode("utf-8") if isinstance(source, str) else source
    normal = bytearray()

    for byte in raw[:WINDOW_BYTES]:
        if 0x41 <= byte <= 0x5A:  # A-Z
            normal.append(byte + 0x20)
        elif byte < 0x20 or byte == 0x7F or byte >= 0x80:
            normal.append(_QUESTION_MARK if byte >= 0x80 else _SPACE)
        else:
            normal.append(byte)

    normal.extend(bytes([_SPACE]) * (WINDOW_BYTES - len(normal)))
    return bytes(normal)


def tokenize(source: bytes | str) -> bytes:
    """Maps text to the ``TOKEN_COUNT`` uint8 ids the emitted graph consumes."""
    normal = normalize_bytes(source)
    tokens = bytearray(byte - 0x1F for byte in normal)  # 1..96
    pairs = normal + bytes([_SPACE])  # sentinel keeps the pair count exact

    for index in range(WINDOW_BYTES):
        tokens.append(BIGRAM_BASE + fnv1a32(pairs[index:index + 2]) % BIGRAM_BUCKETS)

    return bytes(tokens)


def detokenize(ids: Sequence[int]) -> bytes:
    """Inverts the byte channel: character ids back to normal-form bytes.

    Only ids ``1..96`` are invertible; bigram buckets are one-way, which is why
    the round-trip guarantee is stated over the byte channel (see the tests).
    """
    normal = bytearray()
    for index, token in enumerate(ids):
        if not CHARACTER_BASE <= token < BIGRAM_BASE:
            raise ValueError(f"token {index} is not a character id: {token}")
        normal.append(token + 0x1F)
    return bytes(normal)
