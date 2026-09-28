"""Hazard detection baseline evaluation: char n-gram + Logistic Regression.

Produces the confusion matrix and accuracy metrics for the baseline that
the Kiln int8 student is compared against in the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from examples.hazard.graph import HAZARD_CLASSES
from examples.hazard.synthesize import load_corpus, load_held_out

__all__ = ("main",)


def extract_char_ngrams(snippets: list[str], ngram_range: tuple[int, int] = (2, 5)) -> tuple[CountVectorizer, np.ndarray]:
    """Extract character n-grams from code snippets."""
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        min_df=1,
    )
    X = vectorizer.fit_transform(snippets)
    return vectorizer, X


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    """Train a logistic regression classifier."""
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=1337,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_and_report(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_held: np.ndarray,
    y_held: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Evaluate and produce detailed metrics including confusion matrices."""
    test_preds = clf.predict(X_test)
    held_preds = clf.predict(X_held)

    test_acc = accuracy_score(y_test, test_preds)
    held_acc = accuracy_score(y_held, held_preds)

    test_cm = confusion_matrix(y_test, test_preds, labels=list(range(len(class_names))))
    held_cm = confusion_matrix(y_held, held_preds, labels=list(range(len(class_names))))

    per_class_test = test_cm.diagonal() / test_cm.sum(axis=1)
    per_class_held = held_cm.diagonal() / held_cm.sum(axis=1)

    return {
        "test_accuracy": test_acc,
        "held_out_accuracy": held_acc,
        "test_confusion_matrix": test_cm.tolist(),
        "held_out_confusion_matrix": held_cm.tolist(),
        "per_class_test_accuracy": per_class_test.tolist(),
        "per_class_held_out_accuracy": per_class_held.tolist(),
        "class_names": class_names,
        "n_test": len(y_test),
        "n_held_out": len(y_held),
    }


def print_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str) -> None:
    """Pretty print a confusion matrix."""
    print(f"\n{title}")
    print("=" * 80)
    # Header
    print(f"{'':<22}", end="")
    for name in class_names:
        print(f"{name[:12]:>12}", end="")
    print()
    # Rows
    for i, row in enumerate(cm):
        print(f"{class_names[i][:20]:<22}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()


def print_per_class_accuracy(per_class: list[float], class_names: list[str], title: str) -> None:
    """Print per-class accuracy."""
    print(f"\n{title}")
    print("-" * 40)
    for name, acc in zip(class_names, per_class):
        print(f"  {name:<22}: {acc * 100:>5.1f}%")


def main() -> int:
    print("=" * 80)
    print("HAZARD DETECTION BASELINE: char n-gram (2-5) + Logistic Regression")
    print("=" * 80)

    # Load data
    corpus = load_corpus()
    held_out = load_held_out()

    # Split corpus: 70/15/15 (train/val/test) with seed 1337
    # This matches the split in train.py
    import torch
    generator = torch.Generator().manual_seed(1337)
    count = len(corpus)
    order = torch.randperm(count, generator=generator).tolist()
    train_end = int(count * 0.70)
    val_end = train_end + int(count * 0.15)
    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    train_snippets = [corpus[i].snippet for i in train_idx]
    train_labels = [corpus[i].label_id for i in train_idx]

    test_snippets = [corpus[i].snippet for i in test_idx]
    test_labels = [corpus[i].label_id for i in test_idx]

    held_snippets = [s.snippet for s in held_out]
    held_labels = [s.label_id for s in held_out]

    print(f"\nData split (seed 1337):")
    print(f"  Train: {len(train_snippets)} (70%)")
    print(f"  Val:   {len(val_idx)} (15%) - not used for baseline")
    print(f"  Test:  {len(test_snippets)} (15%)")
    print(f"  Held-out: {len(held_snippets)} (real snippets)")

    # Combine train + val for training (as done in train.py for calibration)
    train_val_snippets = train_snippets + [corpus[i].snippet for i in val_idx]
    train_val_labels = train_labels + [corpus[i].label_id for i in val_idx]

    # Extract char n-grams
    print(f"\nExtracting character n-grams (2-5)...")
    vectorizer, X_train_val = extract_char_ngrams(train_val_snippets)
    X_test = vectorizer.transform(test_snippets)
    X_held = vectorizer.transform(held_snippets)

    n_features = X_train_val.shape[1]
    print(f"  Vocabulary size: {n_features} n-grams")
    print(f"  Train+Val shape: {X_train_val.shape}")
    print(f"  Test shape:      {X_test.shape}")
    print(f"  Held-out shape:  {X_held.shape}")

    # Train logistic regression
    print(f"\nTraining Logistic Regression (C=1.0, multinomial)...")
    clf = train_logistic_regression(X_train_val, np.array(train_val_labels))

    # Evaluate
    results = evaluate_and_report(
        clf, X_test, np.array(test_labels), X_held, np.array(held_labels), list(HAZARD_CLASSES)
    )

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Test accuracy (n={results['n_test']}):       {results['test_accuracy'] * 100:.2f}%")
    print(f"Held-out accuracy (n={results['n_held_out']}): {results['held_out_accuracy'] * 100:.2f}%")

    print_confusion_matrix(results["test_confusion_matrix"], results["class_names"], "Test Confusion Matrix")
    print_confusion_matrix(results["held_out_confusion_matrix"], results["class_names"], "Held-out Confusion Matrix")

    print_per_class_accuracy(results["per_class_test_accuracy"], results["class_names"], "Per-Class Test Accuracy")
    print_per_class_accuracy(results["per_class_held_out_accuracy"], results["class_names"], "Per-Class Held-out Accuracy")

    # Feature importance (top n-grams per class)
    print(f"\nTop 10 n-grams per class (by coefficient magnitude):")
    print("-" * 80)
    for i, class_name in enumerate(HAZARD_CLASSES):
        coef = clf.coef_[i]
        top_indices = np.argsort(np.abs(coef))[-10:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        print(f"\n{class_name}:")
        for idx in top_indices:
            print(f"  {feature_names[idx]:<20} coef={coef[idx]:+.4f}")

    # Save results to JSON for evidence
    output = {
        "baseline": "char_ngram_2_5_logistic_regression",
        "vectorizer_vocab_size": int(n_features),
        "train_val_samples": len(train_val_snippets),
        "test_samples": int(results["n_test"]),
        "held_out_samples": int(results["n_held_out"]),
        "test_accuracy": float(results["test_accuracy"]),
        "held_out_accuracy": float(results["held_out_accuracy"]),
        "test_confusion_matrix": results["test_confusion_matrix"],
        "held_out_confusion_matrix": results["held_out_confusion_matrix"],
        "per_class_test_accuracy": results["per_class_test_accuracy"],
        "per_class_held_out_accuracy": results["per_class_held_out_accuracy"],
        "class_names": list(HAZARD_CLASSES),
        "kiln_int8_student_test": 73.3,  # from benchmark
        "kiln_int8_student_held_out": 73.3,
    }

    out_path = Path(__file__).parent / "baseline_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())