from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
import time

from ..retrieval.hybrid import HybridRetriever


CACHE_SCHEMA_VERSION = "v4"


@dataclass(frozen=True)
class CacheMetadata:
    index_version: str
    signature: str
    created_at_epoch: int
    document_count: int
    chunk_count: int
    embedding_mode: str


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
    signature: str | None
    index_version: str | None
    document_count: int
    chunk_count: int
    embedding_mode: str


class IndexCache:
    def __init__(self, cache_dir: Path) -> None:
        """Initializes cache directory and cache file path."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "hybrid_retriever.pkl"

    def load(self, signature: str) -> HybridRetriever | None:
        """Loads retriever bundle if cache exists and signature matches."""
        bundle = self._read_bundle()
        if bundle is None:
            return None
        if bundle.metadata.signature != signature:
            return None
        return bundle.retriever

    def inspect(self, signature: str) -> CacheInspectResult:
        """Returns cache status (`latent`, `stale`, `updated`) plus metadata."""
        bundle = self._read_bundle()
        if bundle is None:
            return CacheInspectResult(
                state="latent",
                matched_signature=False,
                cache_file=self.cache_file,
                created_at_epoch=None,
                signature=None,
                index_version=None,
                document_count=0,
                chunk_count=0,
                embedding_mode="unknown",
            )
        matched = bundle.metadata.signature == signature
        return CacheInspectResult(
            state="updated" if matched else "stale",
            matched_signature=matched,
            cache_file=self.cache_file,
            created_at_epoch=bundle.metadata.created_at_epoch,
            signature=bundle.metadata.signature,
            index_version=bundle.metadata.index_version,
            document_count=bundle.metadata.document_count,
            chunk_count=bundle.metadata.chunk_count,
            embedding_mode=bundle.metadata.embedding_mode,
        )

    def save(self, signature: str, retriever: HybridRetriever) -> None:
        """Serializes retriever + metadata to cache file."""
        embedding_mode = retriever.semantic.mode if hasattr(retriever, "semantic") else "unknown"
        bundle = CacheBundle(
            metadata=CacheMetadata(
                index_version=CACHE_SCHEMA_VERSION,
                signature=signature,
                created_at_epoch=int(time.time()),
                document_count=len(retriever.records),
                chunk_count=len(getattr(retriever, "chunks", [])),
                embedding_mode=embedding_mode,
            ),
            retriever=retriever,
        )
        with self.cache_file.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        """Deletes cache file if present."""
        if self.cache_file.exists():
            self.cache_file.unlink()

    def _read_bundle(self) -> CacheBundle | None:
        """Reads and validates cached bundle, normalizing legacy metadata."""
        if not self.cache_file.exists():
            return None
        try:
            with self.cache_file.open("rb") as handle:
                bundle = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(bundle, CacheBundle):
            return None
        metadata = bundle.metadata
        if not isinstance(metadata, CacheMetadata):
            return None
        try:
            normalized_metadata = self._normalize_metadata(metadata, bundle.retriever)
        except Exception:
            return None
        return CacheBundle(metadata=normalized_metadata, retriever=bundle.retriever)

    def _normalize_metadata(self, metadata: CacheMetadata, retriever: HybridRetriever) -> CacheMetadata:
        """Backfills/normalizes metadata fields for compatibility."""
        signature = getattr(metadata, "signature", "")
        if not signature:
            raise ValueError("Invalid cache metadata: missing signature")

        created_at_epoch = int(getattr(metadata, "created_at_epoch", 0) or 0)
        document_count = int(getattr(metadata, "document_count", len(retriever.records)) or 0)
        chunk_count = int(getattr(metadata, "chunk_count", len(getattr(retriever, "chunks", []))) or 0)
        embedding_mode = str(getattr(metadata, "embedding_mode", getattr(retriever.semantic, "mode", "unknown")) or "unknown")
        index_version = str(getattr(metadata, "index_version", "legacy") or "legacy")

        return CacheMetadata(
            index_version=index_version,
            signature=signature,
            created_at_epoch=created_at_epoch,
            document_count=document_count,
            chunk_count=chunk_count,
            embedding_mode=embedding_mode,
        )


def build_signature(
    excel_source_paths: list[Path],
    embedding_model: str,
    has_openai_key: bool,
    rules_fingerprint: str,
) -> str:
    """Builds deterministic cache signature from sources, model mode, and rules."""
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
