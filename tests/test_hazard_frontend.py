"""M2.4: the C tokenizer and the whole text -> label path, bit-exact.

The M2.0 contract is only deployable if a C frontend reproduces it exactly, so
this suite compiles `examples/hazard/frontend/hazard_tokenizer.c` and compares it
against `examples/hazard/tokenizer.py` over every byte value, then runs the
complete path -- raw snippet -> C tokenizer -> *emitted* C graph -- against the
Python contract plus `kiln.ref`.  It is the Phase 2 analogue of
`tests/test_kws_frontend.py`.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from examples.hazard.graph import DEFAULT_LABELS, build_bag_graph, random_weights
from examples.hazard.tokenizer import (
    BIGRAM_BASE,
    BIGRAM_BUCKETS,
    TOKEN_COUNT,
    VOCAB_SIZE,
    WINDOW_BYTES,
    tokenize,
)
from kiln.backend.c99.emit_c import emit_c
from kiln.ref.ref import run_ref

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "examples" / "hazard" / "frontend"
TOKENIZER_SRC = FRONTEND_DIR / "hazard_tokenizer.c"
PIPELINE_SRC = FRONTEND_DIR / "hazard_pipeline.c"

GCC_FLAGS = ("-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc required for C99 parity")

#: Cases that cover every branch of the normal form plus the window edges.
CASES: tuple[bytes, ...] = (
    b"",
    b"x",
    b"eval(user_input)",
    b"subprocess.run(cmd, shell=True)",
    b"import pickle\npickle.loads(blob)",
    b"os.system('rm -rf /')",
    b"assert user.is_admin\x00\x7f",
    b"\xff\xfe\x80\x81",  # only non-ASCII bytes
    bytes(range(256)),  # every byte value once
    b"A" * 500,  # longer than the window
    b" " * WINDOW_BYTES,  # exactly the window
    b" " * (WINDOW_BYTES + 1),  # one byte over the window
)

_HEX_HEADER = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hazard_tokenizer.h"

#define _MAX_SOURCE 512

static int _decode(const char *hex, uint8_t *out, size_t *out_length) {
    size_t digits = strlen(hex);
    if (digits % 2 != 0 || digits / 2 > _MAX_SOURCE) return 0;
    for (size_t i = 0; i < digits / 2; ++i) {
        char pair[3] = { hex[2 * i], hex[2 * i + 1], '\\0' };
        out[i] = (uint8_t)strtoul(pair, NULL, 16);
    }
    *out_length = digits / 2;
    return 1;
}
"""

_TOKENIZER_MAIN = (
    _HEX_HEADER
    + """
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "constants") == 0) {
        printf("%d %d %d %d %d\\n",
               HAZARD_WINDOW_BYTES, HAZARD_TOKEN_COUNT, HAZARD_VOCAB_SIZE,
               HAZARD_BIGRAM_BASE, HAZARD_BIGRAM_BUCKETS);
        return 0;
    }
    if (argc < 3) return 1;

    uint8_t source[_MAX_SOURCE];
    size_t length = 0;
    if (!_decode(argv[2], source, &length)) return 2;

    uint8_t tokens[HAZARD_TOKEN_COUNT];
    hazard_tokenize(source, length, tokens);
    for (int i = 0; i < HAZARD_TOKEN_COUNT; ++i) {
        printf("%d\\n", (int)tokens[i]);
    }
    return 0;
}
"""
)


_PIPELINE_MAIN = (
    _HEX_HEADER
    + """
#include "hazard_pipeline.h"

int main(int argc, char **argv) {
    if (argc < 2) return 1;

    uint8_t source[_MAX_SOURCE];
    size_t length = 0;
    if (!_decode(argv[1], source, &length)) return 2;

    printf("%d\\n", (int)hazard_predict_source(source, length));
    return 0;
}
"""
)


def _random_cases(count: int = 24) -> list[bytes]:
    """Random byte strings, including empty and longer-than-window ones."""
    generator = random.Random(1337)
    return [
        bytes(generator.randrange(256) for _ in range(generator.randrange(0, 200)))
        for _ in range(count)
    ]


def _build(binary_name: str, source_code: str, extra: tuple[str, ...] = ()) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="hazard_frontend_"))
    source = directory / f"{binary_name}.c"
    binary = directory / f"{binary_name}.exe"
    source.write_text(source_code, encoding="utf-8")

    build = subprocess.run(
        [
            "gcc",
            *GCC_FLAGS,
            f"-I{FRONTEND_DIR}",
            str(source),
            str(TOKENIZER_SRC),
            *extra,
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, f"gcc failed:\n{build.stderr}"

    return binary


@pytest.fixture(scope="module")
def tokenizer_exe() -> Path:
    """The C tokenizer plus a driver that prints the ids of a hex-encoded snippet."""
    return _build("tokenizer_driver", _TOKENIZER_MAIN)


@pytest.fixture(scope="module")
def pipeline_exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Raw snippet -> C tokenizer -> emitted Phase 2 graph -> label, all in C."""
    graph = build_bag_graph()
    weights = random_weights()
    return _build(
        "hazard_pipeline",
        emit_c(graph, weights, runtime_header=None) + _PIPELINE_MAIN,
        extra=(str(PIPELINE_SRC),),
    )


def test_header_constants_match_the_contract(tokenizer_exe: Path) -> None:
    result = subprocess.run(
        [str(tokenizer_exe), "constants"], capture_output=True, text=True, check=True
    )
    window, tokens, vocab, bigram_base, bigram_buckets = (
        int(value) for value in result.stdout.split()
    )

    assert (window, tokens, vocab) == (WINDOW_BYTES, TOKEN_COUNT, VOCAB_SIZE)
    assert (bigram_base, bigram_buckets) == (BIGRAM_BASE, BIGRAM_BUCKETS)


def test_c_tokenizer_matches_the_python_contract(tokenizer_exe: Path) -> None:
    for source in [*CASES, *_random_cases()]:
        result = subprocess.run(
            [str(tokenizer_exe), "tokens", source.hex()],
            capture_output=True,
            text=True,
            check=True,
        )
        compiled = [int(line) for line in result.stdout.split()]

        assert compiled == list(tokenize(source)), source


def test_text_to_label_matches_the_contract(pipeline_exe: Path) -> None:
    graph = build_bag_graph()
    weights = random_weights()

    for source in [*CASES, *_random_cases()]:
        result = subprocess.run(
            [str(pipeline_exe), source.hex()], capture_output=True, text=True, check=True
        )
        compiled = int(result.stdout.strip())
        expected = run_ref(graph, {"tokens": list(tokenize(source))}, weights)["label"][0]

        assert compiled == expected, source
        assert 0 <= compiled < DEFAULT_LABELS
