from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
import time

from ..retrieval.hybrid import HybridRetriever


CACHE_SCHEMA_VERSION = "v3"


@dataclass(frozen=True)
class CacheMetadata:
    signature: str
    created_at_epoch: int


@dataclass(frozen=True)
class CacheBundle:
    metadata: CacheMetadata
    retriever: HybridRetriever


@dataclass(frozen=True)
class CacheInspectResult:
    state: str
    matched_signature: bool
    cache_file: Path
    created_at_epoch: int | None


class IndexCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "hybrid_retriever.pkl"

    def load(self, signature: str) -> HybridRetriever | None:
        bundle = self._read_bundle()
        if bundle is None:
            return None
        if bundle.metadata.signature != signature:
            return None
        return bundle.retriever

    def inspect(self, signature: str) -> CacheInspectResult:
        bundle = self._read_bundle()
        if bundle is None:
            return CacheInspectResult(
                state="latent",
                matched_signature=False,
                cache_file=self.cache_file,
                created_at_epoch=None,
            )
        matched = bundle.metadata.signature == signature
        return CacheInspectResult(
            state="updated" if matched else "latent",
            matched_signature=matched,
            cache_file=self.cache_file,
            created_at_epoch=bundle.metadata.created_at_epoch,
        )

    def save(self, signature: str, retriever: HybridRetriever) -> None:
        bundle = CacheBundle(
            metadata=CacheMetadata(signature=signature, created_at_epoch=int(time.time())),
            retriever=retriever,
        )
        with self.cache_file.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        if self.cache_file.exists():
            self.cache_file.unlink()

    def _read_bundle(self) -> CacheBundle | None:
        if not self.cache_file.exists():
            return None
        try:
            with self.cache_file.open("rb") as handle:
                bundle = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(bundle, CacheBundle):
            return None
        return bundle


def build_signature(
    excel_source_paths: list[Path],
    embedding_model: str,
    has_openai_key: bool,
    rules_fingerprint: str,
) -> str:
    source_parts: list[str] = []
    for source_path in sorted(excel_source_paths, key=lambda p: str(p)):
        stat = source_path.stat()
        source_parts.append(
            "|".join(
                [
                    str(source_path.resolve()),
                    str(stat.st_size),
                    str(stat.st_mtime_ns),
                ]
            )
        )
    payload = "|".join(
        [CACHE_SCHEMA_VERSION, *source_parts, embedding_model, "openai" if has_openai_key else "local", rules_fingerprint]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
