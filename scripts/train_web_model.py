#!/usr/bin/env python
"""Train and export a compact browser-side complaint priority model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_DESCRIPTIONS = {
    1: "단순 문의/정보 요청",
    2: "개인 불편 중심 일반 요청",
    3: "구체 조치가 필요한 일반 민원",
    4: "공공안전/다수 피해 우선 처리",
    5: "생명/재난/안보 등 즉시 대응",
}


def _rounded_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(values.astype(np.float32), digits).tolist()


def load_balanced_sample(input_path: Path, max_per_class: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=["Q_refined", "importance"])
    df = df.rename(columns={"Q_refined": "text"})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].copy()
    df["importance"] = df["importance"].astype(int)

    parts = []
    for label, group in df.groupby("importance", sort=True):
        take = min(len(group), max_per_class)
        parts.append(group.sample(n=take, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def train_model(args: argparse.Namespace):
    data = load_balanced_sample(args.input, args.max_per_class, args.random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["importance"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["importance"],
    )

    n_features = 2 ** args.feature_power
    model = Pipeline(
        steps=[
            (
                "hash",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    n_features=n_features,
                    alternate_sign=False,
                    norm=None,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True, norm="l2")),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=args.alpha,
                    max_iter=args.max_iter,
                    tol=1e-3,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "sample_rows": int(len(data)),
        "max_per_class": int(args.max_per_class),
        "feature_power": int(args.feature_power),
        "n_features": int(n_features),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "label_distribution": {
            str(k): int(v) for k, v in data["importance"].value_counts().sort_index().items()
        },
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5]).tolist(),
    }
    return model, metrics


def export_model(model: Pipeline, metrics: dict, model_path: Path, metrics_path: Path) -> None:
    vectorizer: HashingVectorizer = model.named_steps["hash"]
    tfidf: TfidfTransformer = model.named_steps["tfidf"]
    clf: SGDClassifier = model.named_steps["clf"]

    payload = {
        "schema_version": 1,
        "model_type": "hashing_charwb_tfidf_sgd_logistic",
        "labels": [int(x) for x in clf.classes_],
        "label_descriptions": {str(k): v for k, v in LABEL_DESCRIPTIONS.items()},
        "analyzer": "char_wb",
        "ngram_range": list(vectorizer.ngram_range),
        "n_features": int(vectorizer.n_features),
        "alternate_sign": False,
        "lowercase": False,
        "sublinear_tf": True,
        "norm": "l2",
        "idf": _rounded_list(tfidf.idf_),
        "coef": [_rounded_list(row) for row in clf.coef_],
        "intercept": _rounded_list(clf.intercept_),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "test_rows": metrics["test_rows"],
            "sample_rows": metrics["sample_rows"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/rule_v3_trainable.csv.gz"))
    parser.add_argument("--model-out", type=Path, default=Path("web/model/complaint_priority_model.json"))
    parser.add_argument("--metrics-out", type=Path, default=Path("web/model/metrics.json"))
    parser.add_argument("--max-per-class", type=int, default=80000)
    parser.add_argument("--feature-power", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=18)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metrics = train_model(args)
    export_model(model, metrics, args.model_out, args.metrics_out)
    print(f"saved model: {args.model_out}")
    print(f"saved metrics: {args.metrics_out}")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
