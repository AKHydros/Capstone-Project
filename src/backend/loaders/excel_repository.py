from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..business_rules import (
    categorize_question_labels,
    infer_survey_name,
    infer_wave_year,
    is_valid_question_text,
)
from ..models import QuestionRecord


@dataclass
class ExcelRepository:
    source_paths: list[Path]

    def load_records(self) -> list[QuestionRecord]:
        """Reads Excel sheets, builds value-label mappings, and returns normalized `QuestionRecord` list."""
        records: list[QuestionRecord] = []
        seen_question_ids: set[str] = set()
        for source_path in self.source_paths:
            if not source_path.exists():
                continue

            variable_info = pd.read_excel(source_path, sheet_name="Variable Information")
            variable_values = pd.read_excel(source_path, sheet_name="Variable Values")

            variable_info = variable_info.rename(
                columns={
                    "Variable": "variable",
                    "Position": "position",
                    "Label": "label",
                    "Measurement Level": "measurement_level",
                    "Role": "role",
                    "Column Width": "column_width",
                    "Alignment": "alignment",
                    "Print Format": "print_format",
                    "Write Format": "write_format",
                    "Missing Values": "missing_values",
                }
            )
            variable_values = variable_values.rename(
                columns={
                    "Value": "variable",
                    "Unnamed: 1": "value_code",
                    "Label": "value_label",
                }
            )

            value_map = (
                variable_values[["variable", "value_code", "value_label"]]
                .dropna(subset=["variable", "value_label"])
                .assign(
                    value_text=lambda df: df["value_code"].astype("string").fillna("")
                    + ": "
                    + df["value_label"].astype("string").fillna("")
                )
                .groupby("variable", dropna=False)["value_text"]
                .apply(list)
                .to_dict()
            )

            for row in variable_info.itertuples(index=False):
                variable = str(getattr(row, "variable", "") or "").strip()
                label = str(getattr(row, "label", "") or "").strip()
                if not variable or not is_valid_question_text(label):
                    continue
                if variable in seen_question_ids:
                    continue
                seen_question_ids.add(variable)
                value_labels = value_map.get(variable, [])
                topic_labels, topic_label_sources = categorize_question_labels(label, value_labels)
                measurement_level = _normalize_cell(getattr(row, "measurement_level", "Unknown")) or "Unknown"
                role = _normalize_cell(getattr(row, "role", "Unknown")) or "Unknown"
                context_fields = {
                    "variable": variable,
                    "position": _normalize_cell(getattr(row, "position", "")),
                    "label": label,
                    "measurement_level": measurement_level,
                    "role": role,
                    "column_width": _normalize_cell(getattr(row, "column_width", "")),
                    "alignment": _normalize_cell(getattr(row, "alignment", "")),
                    "print_format": _normalize_cell(getattr(row, "print_format", "")),
                    "write_format": _normalize_cell(getattr(row, "write_format", "")),
                    "missing_values": _normalize_cell(getattr(row, "missing_values", "")),
                }
                context_fields = {key: value for key, value in context_fields.items() if value}

                records.append(
                    QuestionRecord(
                        question_id=variable,
                        question_text=label,
                        measurement_level=measurement_level,
                        role=role,
                        source_file=str(source_path),
                        survey_name=infer_survey_name(variable),
                        wave_year=infer_wave_year(variable),
                        value_labels=value_labels,
                        topic_labels=topic_labels,
                        topic_label_sources=topic_label_sources,
                        context_fields=context_fields,
                    )
                )
        if not records:
            raise FileNotFoundError("No valid Excel source files were found for ingestion.")
        return records


def _normalize_cell(value: object) -> str:
    """Normalize heterogeneous Excel cell values into compact display strings."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
