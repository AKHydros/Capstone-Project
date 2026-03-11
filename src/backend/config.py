from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    excel_source_path: Path
    index_cache_dir: Path



def load_config() -> AppConfig:
    source = os.getenv(
        "EXCEL_SOURCE_PATH",
        "/Users/alexkatzighera/Desktop/Queens MMAI/Capstone/market_research_capstone_draft_1.xlsx",
    )
    cache_dir = os.getenv(
        "INDEX_CACHE_DIR",
        "/Users/alexkatzighera/Documents/Capstone Project/data/cache",
    )
    return AppConfig(excel_source_path=Path(source), index_cache_dir=Path(cache_dir))
