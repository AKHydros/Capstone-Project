from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..business_rules import infer_survey_name, infer_wave_year, is_valid_question_text
from ..models import QuestionRecord


@dataclass
class ExcelRepository:
    source_paths: list[Path]

    def load_records(self) -> list[QuestionRecord]:
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
                    "Label": "label",
                    "Measurement Level": "measurement_level",
                    "Role": "role",
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

                records.append(
                    QuestionRecord(
                        question_id=variable,
                        question_text=label,
                        measurement_level=str(getattr(row, "measurement_level", "") or "Unknown"),
                        role=str(getattr(row, "role", "") or "Unknown"),
                        source_file=str(source_path),
                        survey_name=infer_survey_name(variable),
                        wave_year=infer_wave_year(variable),
                        value_labels=value_map.get(variable, []),
                    )
                )
        if not records:
            raise FileNotFoundError("No valid Excel source files were found for ingestion.")
        return records
