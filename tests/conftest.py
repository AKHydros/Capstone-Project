"""Pytest configuration and shared fixtures for PMG Intelligence test suite.

Adds ``src/`` to ``sys.path`` so all test modules can import ``backend.*``
without needing an editable install.  Also provides reusable fixtures for the
observability store, LLM client mock, and sample records.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensures `from backend.xxx import yyy` resolves against src/ in all test files.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Yields a fresh :class:`~backend.observability.db.ObservabilityStore` backed
    by a temporary SQLite file.  Automatically closed and deleted after the test.
    """
    from backend.observability.db import ObservabilityStore

    db = ObservabilityStore(tmp_path / "test_obs.db")
    yield db
    db.close()


@pytest.fixture()
def mock_llm_client():
    """Returns a :class:`unittest.mock.MagicMock` for ``OpenAIChatClient``.

    Pre-configures ``enabled`` to ``True`` and ``summarize`` to return a
    canned answer so tests that invoke the LLM path do not need a real key.
    """
    from backend.llm.openai_client import OpenAIChatClient

    mock = MagicMock(spec=OpenAIChatClient)
    mock.enabled = True
    mock.summarize.return_value = (
        "**Reasoning:** Using record [PMG22_WAI_q5a] which matches the query.\n"
        "**Answer:** Yes, this is a financial planning question.\n"
        "**Key Details:**\n- Survey: PMG22_WAI\n- Wave: 2022\n"
        "**Suggested Follow-up:** What are the value labels for this question?"
    )
    return mock


@pytest.fixture()
def sample_records():
    """Returns a list of 10 :class:`~backend.models.QuestionRecord` instances
    covering two surveys so retrieval tests have meaningful data to work with.
    """
    from backend.models import QuestionRecord

    surveys = [
        ("PMG22_WAI", "2022"),
        ("PMG18_ROB", "2018"),
    ]
    records = []
    topics = ["Financial Planning", "Demographics", "Trust & Sentiment",
              "Digital Behavior", "Product Ownership"]
    for i in range(10):
        survey_name, wave_year = surveys[i % 2]
        topic = topics[i % len(topics)]
        records.append(
            QuestionRecord(
                question_id=f"{survey_name}_q{i + 1}",
                question_text=f"Sample question {i + 1} about {topic.lower()}",
                survey_name=survey_name,
                wave_year=wave_year,
                topic_labels=[topic],
                topic_label_sources={topic: "Question Text"},
                value_labels=[f"{j}: Option {j}" for j in range(1, 4)],
                measurement_level="Nominal",
                role="Main",
                source_file="test_data.xlsx",
            )
        )
    return records
