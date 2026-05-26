from __future__ import annotations

import json
import gzip
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from physh_tools.physh import load_taxonomy_file


def normalize_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().split())


def sample_f1(predicted: list[str], gold: list[str]) -> float:
    pred_set = {normalize_label(x) for x in predicted if normalize_label(x)}
    gold_set = {normalize_label(x) for x in gold if normalize_label(x)}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return 2.0 * len(pred_set & gold_set) / (len(pred_set) + len(gold_set))


def mean_sample_f1(predicted_rows: list[list[str]], gold_rows: list[list[str]]) -> float:
    if len(predicted_rows) != len(gold_rows):
        raise ValueError("predicted_rows and gold_rows must have the same length")
    if not predicted_rows:
        return 0.0
    return float(np.mean([sample_f1(pred, gold) for pred, gold in zip(predicted_rows, gold_rows)]))


def load_similarity_npz(path: str | Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    labels = [str(x) for x in payload["labels"].tolist()]
    matrix = np.asarray(payload["similarity"], dtype=np.float32)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"Similarity matrix shape {matrix.shape} does not match {len(labels)} labels")
    return labels, matrix


def semantic_precision_recall_f1(
    predicted: list[str],
    gold: list[str],
    *,
    labels: list[str],
    similarity: np.ndarray,
) -> tuple[float, float, float]:
    label_to_idx = {normalize_label(label): idx for idx, label in enumerate(labels)}
    pred_idxs = [label_to_idx[normalize_label(x)] for x in predicted if normalize_label(x) in label_to_idx]
    gold_idxs = [label_to_idx[normalize_label(x)] for x in gold if normalize_label(x) in label_to_idx]
    if not pred_idxs or not gold_idxs:
        return 0.0, 0.0, 0.0

    precision = float(np.mean([np.max(similarity[p, gold_idxs]) for p in pred_idxs]))
    recall = float(np.mean([np.max(similarity[g, pred_idxs]) for g in gold_idxs]))
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def mean_semantic_f1(
    predicted_rows: list[list[str]],
    gold_rows: list[list[str]],
    *,
    labels: list[str],
    similarity: np.ndarray,
) -> dict[str, float]:
    if len(predicted_rows) != len(gold_rows):
        raise ValueError("predicted_rows and gold_rows must have the same length")
    if not predicted_rows:
        return {"semantic_precision": 0.0, "semantic_recall": 0.0, "semantic_f1": 0.0}

    scores = [
        semantic_precision_recall_f1(pred, gold, labels=labels, similarity=similarity)
        for pred, gold in zip(predicted_rows, gold_rows)
    ]
    return {
        "semantic_precision": float(np.mean([x[0] for x in scores])),
        "semantic_recall": float(np.mean([x[1] for x in scores])),
        "semantic_f1": float(np.mean([x[2] for x in scores])),
    }


def _mapping_label(mapping_value: Any) -> str:
    if isinstance(mapping_value, dict):
        return str(mapping_value.get("label", ""))
    return str(mapping_value)


def labels_from_taxonomy_file(name: str) -> list[str]:
    return sorted({_mapping_label(value) for value in load_taxonomy_file(name).values() if _mapping_label(value)})


def labels_from_jsonl_record(record: dict[str, Any], kind: str) -> list[str]:
    """Extract discipline or concept labels from public prediction/gold formats."""
    if kind not in {"disciplines", "concepts"}:
        raise ValueError("kind must be 'disciplines' or 'concepts'")

    candidate_keys = [
        f"predicted_{kind}",
        f"gold_{kind}",
        f"true_{kind}",
        kind,
    ]
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]

    schemes = record.get("classificationSchemes")
    physh = schemes.get("physh") if isinstance(schemes, dict) else None
    if isinstance(physh, dict):
        entries = physh.get(kind)
        if isinstance(entries, list):
            mapping = (
                load_taxonomy_file("disciplines.json")
                if kind == "disciplines"
                else load_taxonomy_file("concepts.json")
            )
            out: list[str] = []
            for entry in entries:
                if isinstance(entry, dict):
                    label = entry.get("label")
                    if label:
                        out.append(str(label))
                        continue
                    entry_id = entry.get("id")
                    if entry_id and str(entry_id) in mapping:
                        out.append(_mapping_label(mapping[str(entry_id)]))
                elif isinstance(entry, str):
                    mapped = mapping.get(entry, mapping.get(_hyphenate_uuid(entry), entry))
                    out.append(_mapping_label(mapped))
            return out

    return []


def _hyphenate_uuid(value: str) -> str:
    text = str(value or "").replace("-", "")
    if len(text) == 32:
        return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"
    return str(value)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows
