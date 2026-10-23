"""M2.0 contract tests: the fixed-point feature contract for Phase 2.

The contract is only useful if it is total (every byte maps), round-trippable
(the byte channel inverts exactly) and reproducible in C later (M2.4), so this
suite pins the byte-level rules, the id ranges, the round trip, the budget
arithmetic of the graph that consumes the ids, and one gcc-compiled check that
runs *text* through the contract and the emitted C to a label.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from examples.hazard.graph import DEFAULT_DIM, DEFAULT_LABELS, build_bag_graph, random_weights
from examples.hazard.tokenizer import (
    BIGRAM_BASE,
    BIGRAM_BUCKETS,
    CHARACTER_BASE,
    CHARACTER_COUNT,
    TOKEN_COUNT,
    VOCAB_SIZE,
    WINDOW_BYTES,
    detokenize,
    fnv1a32,
    normalize_bytes,
    tokenize,
)
from kiln.backend.c99.emit_c import emit_c
from kiln.budget.report import audit_graph
from kiln.ref.ref import run_ref

#: Snippets that exercise every normal-form rule: empty, short, at the window
#: boundary, longer than the window, controls, DEL, non-ASCII and lowercasing.
SNIPPETS: tuple[bytes, ...] = (
    b"",
    b"x = 1",
    b"eval(user_input)",
    b"subprocess.run(cmd, shell=True)",
    b"import pickle\npickle.loads(blob)",
    b"os.system('rm -rf /')",
    b"yaml.load(text, Loader=yaml.Loader)",
    b"tempfile.mktemp()",
    b"assert user.is_admin",
    b"def Eval(Source):\n\treturn eval(Source)\x00\x7f",
    "unicode: λ → ok".encode(),
    b"A" * 500,
)


def test_contract_constants_match_the_ir_limits() -> None:
    assert VOCAB_SIZE == 256  # EMBEDDING addresses its table with uint8
    assert TOKEN_COUNT == 2 * WINDOW_BYTES
    assert (CHARACTER_BASE, CHARACTER_COUNT) == (1, 96)
    assert (BIGRAM_BASE, BIGRAM_BUCKETS) == (97, 159)
    assert BIGRAM_BASE + BIGRAM_BUCKETS == VOCAB_SIZE
    assert BIGRAM_BASE + fnv1a32(b"ab") % BIGRAM_BUCKETS < VOCAB_SIZE


@pytest.mark.parametrize("snippet", SNIPPETS)
def test_normal_form_is_fixed_length_ascii(snippet: bytes) -> None:
    normal = normalize_bytes(snippet)

    assert len(normal) == WINDOW_BYTES
    assert all(0x20 <= byte <= 0x7F for byte in normal)
    assert normalize_bytes(normal) == normal  # idempotent


@pytest.mark.parametrize("snippet", SNIPPETS)
def test_normal_form_rules_independently(snippet: bytes) -> None:
    """Re-derives the documented rules instead of trusting the implementation."""
    normal = normalize_bytes(snippet)
    head = snippet[:WINDOW_BYTES]

    for index, byte in enumerate(head):
        if 0x41 <= byte <= 0x5A:
            expected = byte + 0x20
        elif byte < 0x20 or byte == 0x7F:
            expected = 0x20
        elif byte >= 0x80:
            expected = 0x3F
        else:
            expected = byte
        assert normal[index] == expected, index

    assert normal[len(head) :] == b" " * (WINDOW_BYTES - len(head))


@pytest.mark.parametrize("snippet", SNIPPETS)
def test_tokenize_ids_stay_inside_the_contract(snippet: bytes) -> None:
    tokens = tokenize(snippet)

    assert len(tokens) == TOKEN_COUNT
    assert all(1 <= token < VOCAB_SIZE for token in tokens)  # id 0 never emitted
    assert all(CHARACTER_BASE <= token < BIGRAM_BASE for token in tokens[:WINDOW_BYTES])
    assert all(BIGRAM_BASE <= token < VOCAB_SIZE for token in tokens[WINDOW_BYTES:])


@pytest.mark.parametrize("snippet", SNIPPETS)
def test_round_trip_on_the_byte_channel(snippet: bytes) -> None:
    normal = normalize_bytes(snippet)
    tokens = tokenize(snippet)

    assert detokenize(tokens[:WINDOW_BYTES]) == normal
    assert normalize_bytes(detokenize(tokens[:WINDOW_BYTES])) == normal
    assert tokenize(normal) == tokens  # a function of the normal form alone


def test_bigram_channel_pairs_each_byte_with_its_successor() -> None:
    normal = normalize_bytes(b"ab")
    tokens = tokenize(b"ab")
    pairs = normal + b" "  # the sentinel keeps the pair count exact

    assert len(pairs) == WINDOW_BYTES + 1
    for index in range(WINDOW_BYTES):
        expected = BIGRAM_BASE + fnv1a32(pairs[index : index + 2]) % BIGRAM_BUCKETS
        assert tokens[WINDOW_BYTES + index] == expected


def test_normalisation_makes_case_and_controls_irrelevant() -> None:
    assert tokenize(b"Eval\tSource") == tokenize(b"eval source")
    assert tokenize(b"Pickle.Loads ") == tokenize(b"pickle.loads ")
    assert tokenize(b"os.system\x00(x)") == tokenize(b"os.system (x)")
    assert tokenize("A") == tokenize(b"A")  # str and bytes agree
    assert tokenize(b"eval(1)") == tokenize(b"eval(1)")  # deterministic


def test_truncation_keeps_the_first_window() -> None:
    long = b"pickle.loads(" + b"x" * 500

    assert normalize_bytes(long) == normalize_bytes(long[:WINDOW_BYTES])
    assert tokenize(long) == tokenize(long[:WINDOW_BYTES])


def test_window_fits_the_arena_budget() -> None:
    """The M2.0 window is what the 2 KB budget buys (ROADMAP.md Phase 2)."""
    report = audit_graph(build_bag_graph(), random_weights())

    assert (DEFAULT_DIM, DEFAULT_LABELS) == (8, 6)
    assert report.arena_ram_bytes == TOKEN_COUNT * (DEFAULT_DIM + DEFAULT_LABELS)
    assert report.arena_ram_bytes < 2048
    assert report.total_rom_bytes < 40960


HARNESS = """
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {{
    if (argc < {length} + 1) return 1;
    uint8_t input[{length}];
    for (int i = 0; i < {length}; ++i) input[i] = (uint8_t)atoi(argv[i + 1]);
    printf("%d\\n", (int)kiln_predict(input, {length}));
    return 0;
}}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc required for the text-to-C check")
def test_text_to_emitted_c_label_is_bit_exact() -> None:
    """M2.4 preview: text -> contract ids -> emitted C == kiln.ref, bit for bit."""
    graph = build_bag_graph()
    weights = random_weights()
    samples = [list(tokenize(snippet)) for snippet in SNIPPETS]

    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "hazard.c"
        binary = Path(tmpdir) / "hazard.exe"
        source.write_text(
            emit_c(graph, weights, runtime_header=None) + HARNESS.format(length=TOKEN_COUNT),
            encoding="utf-8",
        )
        build = subprocess.run(
            [
                "gcc",
                "-std=c99",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(source),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr

        compiled = [
            int(
                subprocess.run(
                    [str(binary), *[str(token) for token in sample]],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            for sample in samples
        ]

    reference = [run_ref(graph, {"tokens": sample}, weights)["label"][0] for sample in samples]

    assert compiled == reference
    assert all(0 <= label < DEFAULT_LABELS for label in compiled)
