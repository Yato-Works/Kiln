"""Phase 2 M2.3: the equal-budget hazard showdown (G3, G4, G5).

Two commands reproduce the whole report::

    python -m examples.hazard.train        # retrain + bit-exact-verify the student
    python -m examples.hazard.benchmark    # this showdown

Models (all fitted on the same 210-sample train split only; G5):

1. **Kiln int8 student** - the emitted C artifact (gcc, ``predict_int``
   fallback), 2,096 B ROM / 1,344 B arena under the M2.0 token contract.
2. **fastText** - supervised bag-of-words at ``dim 8`` with subword n-grams
   switched off: the same hypothesis class as the student (a linear model over
   a token bag) but at the corpus' own vocabulary size.  Its footprint is
   measured, so G4's equal-budget claim ("<= 40 KB at int8") is checked.
3. **Rules baseline** - the ordered regex allow-list control G3 asks for;
   zero fitting, first match wins, benign otherwise.
4. **LogisticRegression over char n-grams** - the sklearn reference point;
   kept as *context*: it is expected to blow the 40 KB envelope.

Accuracy is reported with Wilson 95% confidence intervals on the frozen test
split (45) and on the real, never-synthesized held-out set (60).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from typing import Any, Final, Sequence

from examples.hazard.graph import DEFAULT_DIM, DEFAULT_LABELS, HAZARD_CLASSES, build_bag_graph
from examples.hazard.synthesize import load_corpus, load_held_out
from examples.hazard.tokenizer import tokenize
from examples.hazard.train import accuracy, load_student, predict_c, predict_int, split_indices
from kiln.budget.report import audit_graph

#: Ordered control rules: first match wins, benign if nothing matches.
RULES: Final[tuple[tuple[str, str], ...]] = (
    ("dynamic_exec", r"\b(eval|exec|compile|__import__)\s*\("),
    ("deserialization", r"\b(pickle|marshal|shelve)\b|\byaml\.load\b|\bloads?\s*\("),
    ("shell_injection", r"\bos\.system\b|\bsubprocess\b|\bshell\s*=\s*True\b|\bpopen\b|\bbash\b|\bsh -c\b"),
    ("tempfile", r"\btempfile\b|\bmktemp\b|\b/tmp/|\bmkstemp\b"),
    ("unsafe_validation", r"\bassert\b"),
)

#: fastText configuration, fixed a priori (no validation or test peeking).
FASTTEXT_CONFIG: Final[dict[str, Any]] = {
    "dim": DEFAULT_DIM,
    "epoch": 50,
    "lr": 0.5,
    "wordNgrams": 1,
    "minn": 0,
    "maxn": 0,
    "minCount": 1,
    "loss": "softmax",
    "thread": 1,
    "verbose": 0,
}

#: Wilson score interval (95%), the CI G5 asks for.
_WILSON_Z: Final = 1.96


def wilson_interval(hits: int, total: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """Percent ``(low, high)`` Wilson score interval for ``hits / total``."""
    if total == 0:
        return (0.0, 0.0)
    proportion = hits / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return (100.0 * max(0.0, centre - margin), 100.0 * min(1.0, centre + margin))


def hits_of(predictions: Sequence[int], labels: Sequence[int]) -> int:
    """Number of exact matches (the input ``wilson_interval`` wants)."""
    return sum(int(predicted) == int(gold) for predicted, gold in zip(predictions, labels))


def format_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> str:
    """``73.3% [58.1, 84.4]`` for a prediction vector against gold labels."""
    hits = hits_of(predictions, labels)
    low, high = wilson_interval(hits, len(labels))
    return f"{accuracy(predictions, labels):.1f}% [{low:.1f}, {high:.1f}]"


def rules_predict(text: str) -> int:
    """The G3 control: ordered regex allow-list, first match wins."""
    for label, pattern in RULES:
        if re.search(pattern, text):
            return HAZARD_CLASSES.index(label)
    return HAZARD_CLASSES.index("benign")


def _import_fasttext() -> Any | None:
    try:
        import fasttext  # noqa: PLC0415 - optional dependency, checked at call time
    except ImportError:
        return None
    return fasttext


def train_fasttext(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
) -> tuple[Any, int]:
    """Fits the fixed-config fastText classifier; returns ``(model, vocab)``."""
    fasttext = _import_fasttext()
    if fasttext is None:
        raise RuntimeError("fasttext is not installed (pip install fasttext-wheel)")
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = f"{tmpdir}/train.txt"
        lines = [
            f"__label__{HAZARD_CLASSES[label]} {text.replace(chr(10), ' ')}"
            for text, label in zip(train_texts, train_labels)
        ]
        with open(train_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        model = fasttext.train_supervised(input=train_path, **FASTTEXT_CONFIG)
    return model, len(model.get_words())


def predict_fasttext(model: Any, texts: Sequence[str]) -> list[int]:
    """fastText top-1 -> frozen taxonomy index."""
    cleaned = [text.replace("\n", " ") for text in texts]
    labels, _ = model.predict(cleaned)
    predictions: list[int] = []
    for batch in labels:
        name = batch[0].removeprefix("__label__")
        predictions.append(HAZARD_CLASSES.index(name))
    return predictions


def _student_footprint(student: dict[str, Any]) -> str:
    report = audit_graph(build_bag_graph(), {"w_emb": student["w_emb"], "w_mm": student["w_mm"]})
    return f"ROM {report.total_rom_bytes:,} B; arena {report.arena_ram_bytes:,} B"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 M2.3: equal-budget hazard showdown")
    parser.add_argument("--no-gcc", action="store_true", help="use the int forward instead of emitted C")
    args = parser.parse_args(argv)

    # ---- frozen data: split, features, labels ----
    corpus = load_corpus()
    held_out = load_held_out()
    texts = [sample.snippet for sample in corpus]
    labels = [sample.label_id for sample in corpus]
    held_texts = [sample.snippet for sample in held_out]
    held_labels = [sample.label_id for sample in held_out]
    train_index, val_index, test_index = split_indices(len(corpus))
    train_texts = [texts[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_texts = [texts[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]
    print(
        f"train {len(train_texts)} (val {len(val_index)} for the student only) / "
        f"test {len(test_texts)} / held-out {len(held_texts)}"
    )

    rows: list[dict[str, str]] = []

    # ---- 1. Kiln int8 student: emitted C when gcc is present ----
    student = load_student()
    w_dict = {"w_emb": student["w_emb"], "w_mm": student["w_mm"]}
    graph = build_bag_graph()
    corpus_rows = [list(tokenize(sample.snippet)) for sample in corpus]
    held_rows = [list(tokenize(sample.snippet)) for sample in held_out]
    quant = graph.tensors["logits"].quant
    if args.no_gcc:
        student_all = predict_int(w_dict, corpus_rows + held_rows, quant.multiplier, quant.shift)
        student_source = "int forward"
    else:
        student_all = predict_c(graph, w_dict, corpus_rows + held_rows)
        student_source = "emitted C"
        if student_all is None:  # gcc missing
            student_all = predict_int(w_dict, corpus_rows + held_rows, quant.multiplier, quant.shift)
            student_source = "int forward (no gcc)"
    student_test = [student_all[i] for i in test_index]
    student_held = student_all[len(corpus):]
    rows.append(
        {
            "name": f"Kiln int8 student ({student_source})",
            "footprint": _student_footprint(student),
            "test": format_accuracy(student_test, test_labels),
            "held": format_accuracy(student_held, held_labels),
            "test_hits": str(hits_of(student_test, test_labels)),
            "held_hits": str(hits_of(student_held, held_labels)),
        }
    )

    # ---- 2. fastText at matched dim, footprint measured (G4) ----
    fasttext = _import_fasttext()
    if fasttext is None:
        rows.append({"name": "fastText (dim 8, no subwords)", "footprint": "SKIPPED (pip install fasttext-wheel)", "test": "-", "held": "-", "test_hits": "0", "held_hits": "0"})
    else:
        model, vocab = train_fasttext(train_texts, train_labels)
        ft_test = predict_fasttext(model, test_texts)
        ft_held = predict_fasttext(model, held_texts)
        words = model.get_words()
        params_i8 = (vocab + DEFAULT_LABELS) * FASTTEXT_CONFIG["dim"]
        dictionary = sum(len(word) + 1 for word in words)
        footprint = (
            f"{vocab} words x dim {FASTTEXT_CONFIG['dim']}: "
            f"{(4 * params_i8 + dictionary) / 1024:.1f} KB f32, "
            f"{(params_i8 + dictionary) / 1024:.1f} KB int8"
        )
        if params_i8 + dictionary > 40960:
            footprint += " OVER BUDGET"
        rows.append(
            {
                "name": "fastText (dim 8, no subwords)",
                "footprint": footprint,
                "test": format_accuracy(ft_test, test_labels),
                "held": format_accuracy(ft_held, held_labels),
                "test_hits": str(hits_of(ft_test, test_labels)),
                "held_hits": str(hits_of(ft_held, held_labels)),
            }
        )

    # ---- 3. rules baseline: the G3 control, zero fitting ----
    rules_test = [rules_predict(text) for text in test_texts]
    rules_held = [rules_predict(text) for text in held_texts]
    rows.append(
        {
            "name": "Rules baseline (regex control)",
            "footprint": f"{len(json.dumps(RULES))} B of regex source",
            "test": format_accuracy(rules_test, test_labels),
            "held": format_accuracy(rules_held, held_labels),
            "test_hits": str(hits_of(rules_test, test_labels)),
            "held_hits": str(hits_of(rules_held, held_labels)),
        }
    )

    # ---- 4. sklearn char n-gram logistic regression (context row) ----
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
    except ImportError:
        rows.append(
            {
                "name": "LogisticRegression (char 2-5)",
                "footprint": "SKIPPED (pip install scikit-learn)",
                "test": "-",
                "held": "-",
                "test_hits": "0",
                "held_hits": "0",
            }
        )
    else:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5))),
                ("clf", LogisticRegression(max_iter=2000, random_state=1337)),
            ]
        )
        pipeline.fit(train_texts, train_labels)
        lr_test = [int(value) for value in pipeline.predict(test_texts)]
        lr_held = [int(value) for value in pipeline.predict(held_texts)]
        features = len(pipeline.named_steps["tfidf"].vocabulary_)
        # idf vector + (labels x features) coefficients, float32
        lr_bytes = (DEFAULT_LABELS + 1) * features * 4
        footprint = f"{features} char n-grams: {lr_bytes / 1024:.0f} KB f32"
        if lr_bytes > 40960:
            footprint += " OVER BUDGET"
        rows.append(
            {
                "name": "LogisticRegression (char 2-5)",
                "footprint": footprint,
                "test": format_accuracy(lr_test, test_labels),
                "held": format_accuracy(lr_held, held_labels),
                "test_hits": str(hits_of(lr_test, test_labels)),
                "held_hits": str(hits_of(lr_held, held_labels)),
            }
        )

    # ---- report ----
    print("\n" + "=" * 124)
    print(f"PHASE 2 M2.3 HAZARD SHOWDOWN (frozen test n={len(test_labels)}, real held-out n={len(held_labels)}, Wilson 95% CI)")
    print("=" * 124)
    header = f"{'model':<33} {'footprint':<45} {'test':<21} {'held-out':<21}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['name']:<33} {row['footprint']:<45} {row['test']:<21} {row['held']:<21}")
    print()

    result = {
        "models": {
            row["name"]: {"test": row["test"], "held_out": row["held"], "test_hits": row["test_hits"], "held_hits": row["held_hits"], "footprint": row["footprint"]}
            for row in rows
        },
        "n_test": len(test_labels),
        "n_held_out": len(held_labels),
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

