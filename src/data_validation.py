"""
data_validation.py — Dataset validation and cleaning utilities.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    missing_text: int = 0
    missing_label: int = 0
    empty_text: int = 0
    short_text: int = 0          # < 10 chars after strip
    invalid_labels: int = 0
    class_counts: dict = field(default_factory=dict)
    unique_texts: int = 0
    label_column: str = ""
    text_column: str = ""
    label_mapping: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_valid: bool = False

    @property
    def class_balance_ratio(self) -> float:
        if not self.class_counts or len(self.class_counts) < 2:
            return 0.0
        counts = list(self.class_counts.values())
        return min(counts) / max(counts)

    @property
    def short_text_count(self) -> int:
        return self.short_text


MIN_SAMPLES = 20
SHORT_TEXT_THRESHOLD = 10


def validate_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_mapping: dict[Any, str] | None = None,  # e.g. {0: "FAKE", 1: "REAL"}
    min_samples_per_class: int = MIN_SAMPLES,
) -> ValidationReport:
    """Run all validation checks and return a ValidationReport."""
    if label_mapping is None:
        if label_col in df.columns:
            uniques = df[label_col].dropna().unique()
            label_mapping = {val: str(val).strip().upper() for val in uniques}
        else:
            label_mapping = {}

    report = ValidationReport(
        total_rows=len(df),
        text_column=text_col,
        label_column=label_col,
        label_mapping=label_mapping,
    )

    # ── Column existence ──
    if text_col not in df.columns:
        report.errors.append(f"Text column '{text_col}' not found in dataset.")
        return report
    if label_col not in df.columns:
        report.errors.append(f"Label column '{label_col}' not found in dataset.")
        return report

    texts  = df[text_col]
    labels = df[label_col]

    # ── Missing values ──
    report.missing_text  = int(texts.isna().sum())
    report.missing_label = int(labels.isna().sum())

    # ── Duplicates ──
    report.duplicate_rows = int(df.duplicated(subset=[text_col]).sum())

    # ── Empty / very short text ──
    non_null_texts = texts.dropna().astype(str).str.strip()
    report.empty_text = int((non_null_texts == "").sum())
    report.short_text = int((non_null_texts.str.len() < SHORT_TEXT_THRESHOLD).sum())
    report.unique_texts = int(non_null_texts.nunique())

    # ── Label validation ──
    allowed_raw = set(label_mapping.keys())
    # Try both string and numeric matching
    raw_labels = labels.dropna()
    def _matches(v) -> bool:
        return v in allowed_raw or str(v) in {str(k) for k in allowed_raw}
    invalid_mask = ~raw_labels.apply(_matches)
    report.invalid_labels = int(invalid_mask.sum())
    if report.invalid_labels > 0:
        unique_invalid = raw_labels[invalid_mask].unique()[:10]
        report.errors.append(
            f"{report.invalid_labels} rows have unrecognised label values: {list(unique_invalid)}"
        )

    # ── Class distribution (after mapping) ──
    def _map_label(v):
        if v in label_mapping:
            return label_mapping[v]
        sv = str(v)
        for k, mapped in label_mapping.items():
            if str(k) == sv:
                return mapped
        return None

    mapped = raw_labels.apply(_map_label).dropna()
    report.class_counts = mapped.value_counts().to_dict()

    # ── Warnings ──
    if report.missing_text > 0:
        report.warnings.append(f"{report.missing_text} rows have missing text — they will be excluded from training.")
    if report.missing_label > 0:
        report.warnings.append(f"{report.missing_label} rows have missing labels — they will be excluded.")
    if report.duplicate_rows > 0:
        report.warnings.append(f"{report.duplicate_rows} duplicate articles detected (same text column).")
    if report.short_text > 0:
        report.warnings.append(f"{report.short_text} articles are very short (< {SHORT_TEXT_THRESHOLD} chars).")

    for cls, cnt in report.class_counts.items():
        if cnt < min_samples_per_class:
            report.errors.append(
                f"Class '{cls}' has only {cnt} samples — minimum required is {min_samples_per_class}."
            )

    if report.class_balance_ratio < 0.2 and len(report.class_counts) >= 2:
        report.warnings.append(
            f"Severe class imbalance detected (ratio {report.class_balance_ratio:.2f}). "
            "Consider using class_weight='balanced' during training."
        )

    if len(report.class_counts) < 2:
        report.errors.append("Dataset must contain at least two distinct classes.")

    # ── Valid row count ──
    invalid_mask_full = (
        df[text_col].isna() |
        df[label_col].isna() |
        (df[text_col].astype(str).str.strip() == "")
    )
    report.invalid_rows = int(invalid_mask_full.sum())
    report.valid_rows   = report.total_rows - report.invalid_rows

    report.is_valid = len(report.errors) == 0 and report.valid_rows >= min_samples_per_class * 2
    return report


def clean_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_mapping: dict | None = None,
    drop_duplicates: bool = True,
    drop_short: bool = False,
    short_threshold: int = SHORT_TEXT_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply the confirmed cleaning steps to the dataset.
    Returns a cleaned copy — does NOT modify the original.
    """
    if label_mapping is None:
        if label_col in df.columns:
            uniques = df[label_col].dropna().unique()
            label_mapping = {val: str(val).strip().upper() for val in uniques}
        else:
            label_mapping = {}

    cleaned = df.copy()

    # Drop rows with missing text or label
    cleaned = cleaned.dropna(subset=[text_col, label_col])

    # Drop empty text
    cleaned = cleaned[cleaned[text_col].astype(str).str.strip() != ""]

    # Map labels
    def _map(v):
        if v in label_mapping:
            return label_mapping[v]
        sv = str(v)
        for k, mapped in label_mapping.items():
            if str(k) == sv:
                return mapped
        return None

    cleaned[label_col] = cleaned[label_col].apply(_map)
    cleaned = cleaned.dropna(subset=[label_col])

    # Keep only rows with valid mapped labels
    valid_labels = set(label_mapping.values())
    cleaned = cleaned[cleaned[label_col].isin(valid_labels)]

    # Drop duplicates
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates(subset=[text_col])

    # Drop very short text
    if drop_short:
        cleaned = cleaned[cleaned[text_col].astype(str).str.strip().str.len() >= short_threshold]

    return cleaned.reset_index(drop=True)
