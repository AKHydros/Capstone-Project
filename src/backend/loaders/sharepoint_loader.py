"""sharepoint_loader.py — SharePoint document connector for the PMG capstone chatbot.

Downloads ``.xlsx`` and ``.docx`` files from a SharePoint document library into
the local ``data/user_uploads/`` directory so they are picked up by the existing
``ExcelRepository`` and ``SurveyPromptLoader`` ingestion pipeline on the next
cache rebuild.

Authentication supports two modes (selected by ``SHAREPOINT_AUTH_MODE``):

* ``"client_credentials"`` (default / production) — uses an Azure AD app
  registration with ``client_id`` + ``client_secret``.  Requires the app to
  have *Sites.Read.All* (or broader) Graph API permission.
* ``"device_flow"`` (development) — prompts the user to authenticate via a
  browser code-flow.  Useful when running locally without a registered app
  secret.

The connector is intentionally dependency-light.  It only imports the optional
``msal`` and ``requests`` packages at *call time*, not at import time, so the
rest of the application starts normally when these packages are absent.  A
clear ``ImportError`` with install instructions is raised only when the user
actually tries to connect.

Environment Variables
---------------------
SHAREPOINT_TENANT_ID
    Azure AD tenant ID (GUID).
SHAREPOINT_CLIENT_ID
    Azure AD application (client) ID.
SHAREPOINT_CLIENT_SECRET
    Client secret for ``client_credentials`` flow.  Leave empty for
    ``device_flow``.
SHAREPOINT_SITE_URL
    Full SharePoint site URL, e.g.
    ``https://contoso.sharepoint.com/sites/ResearchData``.
SHAREPOINT_LIBRARY_PATH
    Server-relative path of the document library to scan, e.g.
    ``/sites/ResearchData/Shared Documents/Survey Dictionaries``.
SHAREPOINT_AUTH_MODE
    ``"client_credentials"`` (default) or ``"device_flow"``.
SHAREPOINT_FILE_EXTENSIONS
    Comma-separated list of extensions to download, default ``".xlsx,.docx"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharePointConfig:
    """Immutable configuration for the SharePoint connector.

    All fields are read from environment variables when
    :func:`SharePointConfig.from_env` is called.  They can also be supplied
    directly for testing.

    Attributes
    ----------
    tenant_id:
        Azure AD tenant GUID.
    client_id:
        Azure AD application (client) ID.
    client_secret:
        Client secret — empty string selects ``device_flow`` authentication.
    site_url:
        Full SharePoint site URL.
    library_path:
        Server-relative path of the document library folder to scan.
    auth_mode:
        ``"client_credentials"`` or ``"device_flow"``.
    file_extensions:
        Tuple of lowercase file extensions to download, e.g.
        ``(".xlsx", ".docx")``.
    """

    tenant_id: str
    client_id: str
    client_secret: str
    site_url: str
    library_path: str
    auth_mode: str = "client_credentials"
    file_extensions: tuple[str, ...] = (".xlsx", ".docx")

    @classmethod
    def from_env(cls) -> "SharePointConfig":
        """Build a ``SharePointConfig`` from environment variables.

        Returns
        -------
        SharePointConfig
            Populated from the ``SHAREPOINT_*`` environment variables.
            Missing variables default to empty strings / the defaults above.
        """
        raw_ext = os.getenv("SHAREPOINT_FILE_EXTENSIONS", ".xlsx,.docx")
        extensions = tuple(
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in raw_ext.split(",")
            if ext.strip()
        )
        return cls(
            tenant_id=os.getenv("SHAREPOINT_TENANT_ID", ""),
            client_id=os.getenv("SHAREPOINT_CLIENT_ID", ""),
            client_secret=os.getenv("SHAREPOINT_CLIENT_SECRET", ""),
            site_url=os.getenv("SHAREPOINT_SITE_URL", ""),
            library_path=os.getenv("SHAREPOINT_LIBRARY_PATH", ""),
            auth_mode=os.getenv("SHAREPOINT_AUTH_MODE", "client_credentials"),
            file_extensions=extensions,
        )

    def is_configured(self) -> bool:
        """Return ``True`` when the minimum required fields are present.

        A connection attempt will fail without ``tenant_id``, ``client_id``,
        and ``site_url``.  ``client_secret`` is only required for the
        ``client_credentials`` flow.

        Returns
        -------
        bool
        """
        if not (self.tenant_id and self.client_id and self.site_url):
            return False
        if self.auth_mode == "client_credentials" and not self.client_secret:
            return False
        return True


# ---------------------------------------------------------------------------
# Download result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SharePointSyncResult:
    """Summary of a SharePoint sync operation.

    Attributes
    ----------
    downloaded:
        Paths of files that were newly downloaded.
    skipped:
        File names that were skipped (already up-to-date or unsupported type).
    errors:
        ``(filename, error_message)`` pairs for files that failed.
    auth_mode:
        Authentication mode used (``"client_credentials"`` or ``"device_flow"``).
    """

    downloaded: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    auth_mode: str = ""

    @property
    def success(self) -> bool:
        """``True`` when at least one file was downloaded without errors."""
        return bool(self.downloaded) and not self.errors

    @property
    def summary(self) -> str:
        """Human-readable one-line summary suitable for UI toasts.

        Returns
        -------
        str
            E.g. ``"SharePoint sync: 3 downloaded, 1 skipped, 0 errors"``.
        """
        return (
            f"SharePoint sync: {len(self.downloaded)} downloaded, "
            f"{len(self.skipped)} skipped, {len(self.errors)} errors."
        )


# ---------------------------------------------------------------------------
# Main connector class
# ---------------------------------------------------------------------------

class SharePointLoader:
    """Downloads survey files from a SharePoint document library.

    Uses the Microsoft Graph API via MSAL for authentication.  Only ``.xlsx``
    and ``.docx`` files (configurable via ``SHAREPOINT_FILE_EXTENSIONS``) are
    downloaded; all others are silently skipped.

    Parameters
    ----------
    config:
        :class:`SharePointConfig` instance.  When *None*, the config is loaded
        from environment variables via :meth:`SharePointConfig.from_env`.
    downloads_dir:
        Directory where downloaded files are saved.  Defaults to
        ``data/user_uploads`` relative to the current working directory.

    Examples
    --------
    Basic usage (client-credentials flow)::

        loader = SharePointLoader()
        result = loader.sync()
        print(result.summary)

    Device-flow for local development (opens a browser prompt)::

        os.environ["SHAREPOINT_AUTH_MODE"] = "device_flow"
        loader = SharePointLoader()
        result = loader.sync()
    """

    _GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    _GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]

    def __init__(
        self,
        config: SharePointConfig | None = None,
        downloads_dir: Path | None = None,
    ) -> None:
        """Initialise the loader with optional config and download directory.

        Parameters
        ----------
        config:
            Pre-built :class:`SharePointConfig`.  Falls back to
            :meth:`SharePointConfig.from_env` when *None*.
        downloads_dir:
            Target directory for downloaded files.  Created if absent.
        """
        self.config = config or SharePointConfig.from_env()
        self.downloads_dir = downloads_dir or (Path.cwd() / "data" / "user_uploads")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate_config(self) -> tuple[bool, str]:
        """Check whether the current config is sufficient to attempt a connection.

        Returns
        -------
        tuple[bool, str]
            ``(True, "ok")`` when config is valid; ``(False, reason)`` otherwise.
        """
        if not self.config.tenant_id:
            return False, "SHAREPOINT_TENANT_ID is not set."
        if not self.config.client_id:
            return False, "SHAREPOINT_CLIENT_ID is not set."
        if not self.config.site_url:
            return False, "SHAREPOINT_SITE_URL is not set."
        if self.config.auth_mode == "client_credentials" and not self.config.client_secret:
            return False, "SHAREPOINT_CLIENT_SECRET is required for client_credentials auth mode."
        return True, "ok"

    def sync(self) -> SharePointSyncResult:
        """Download all supported files from the configured SharePoint library.

        Skips files whose local copy already exists with the same size as the
        remote file (lightweight freshness check — no ETag or checksum).

        Returns
        -------
        SharePointSyncResult
            Summary of downloaded, skipped, and errored files.

        Raises
        ------
        ImportError
            When ``msal`` or ``requests`` are not installed.
        RuntimeError
            When config validation fails or authentication cannot obtain a token.
        """
        self._require_dependencies()
        valid, reason = self.validate_config()
        if not valid:
            raise RuntimeError(f"SharePoint config error: {reason}")

        token = self._acquire_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        result = SharePointSyncResult(auth_mode=self.config.auth_mode)
        site_id = self._resolve_site_id(headers)
        items = self._list_drive_items(site_id, headers)

        for item in items:
            name: str = item.get("name", "")
            suffix = Path(name).suffix.lower()
            if suffix not in self.config.file_extensions:
                result.skipped.append(name)
                continue
            try:
                self._download_item(item, headers, result)
            except Exception as exc:  # noqa: BLE001
                result.errors.append((name, str(exc)))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_dependencies(self) -> None:
        """Raise a clear ImportError when optional dependencies are missing.

        Raises
        ------
        ImportError
            When ``msal`` or ``requests`` are not installed, with install
            instructions included in the message.
        """
        missing: list[str] = []
        try:
            import msal  # noqa: F401
        except ImportError:
            missing.append("msal")
        try:
            import requests  # noqa: F401
        except ImportError:
            missing.append("requests")
        if missing:
            pkgs = " ".join(missing)
            raise ImportError(
                f"SharePoint connector requires: {pkgs}\n"
                f"Install with:  pip install {pkgs}"
            )

    def _acquire_token(self) -> str:
        """Obtain a Microsoft Graph Bearer token via MSAL.

        Uses ``client_credentials`` flow by default, falls back to
        ``device_flow`` when ``SHAREPOINT_AUTH_MODE=device_flow`` or when
        ``client_secret`` is absent.

        Returns
        -------
        str
            A valid Bearer access token.

        Raises
        ------
        RuntimeError
            When MSAL cannot obtain a token (wrong credentials, expired secret,
            tenant mismatch, etc.).
        """
        import msal  # imported lazily to avoid hard dependency

        authority = f"https://login.microsoftonline.com/{self.config.tenant_id}"

        if self.config.auth_mode == "device_flow":
            app = msal.PublicClientApplication(
                client_id=self.config.client_id,
                authority=authority,
            )
            flow = app.initiate_device_flow(scopes=self._GRAPH_SCOPE)
            if "user_code" not in flow:
                raise RuntimeError(f"Could not initiate device flow: {flow.get('error_description', 'unknown error')}")
            # The caller (UI) is responsible for surfacing flow["message"] to the user.
            print(flow["message"])  # visible in terminal / Streamlit server logs
            token_response = app.acquire_token_by_device_flow(flow)
        else:
            app = msal.ConfidentialClientApplication(
                client_id=self.config.client_id,
                client_credential=self.config.client_secret,
                authority=authority,
            )
            token_response = app.acquire_token_for_client(scopes=self._GRAPH_SCOPE)

        if "access_token" not in token_response:
            err = token_response.get("error_description", token_response.get("error", "unknown"))
            raise RuntimeError(f"SharePoint authentication failed: {err}")

        return str(token_response["access_token"])

    def _resolve_site_id(self, headers: dict[str, str]) -> str:
        """Resolve the Graph API site ID from the configured SharePoint site URL.

        Parameters
        ----------
        headers:
            Authorised HTTP headers including the Bearer token.

        Returns
        -------
        str
            The Graph site ID (used in subsequent item-listing calls).

        Raises
        ------
        RuntimeError
            When the site cannot be found or the API returns an error.
        """
        import requests  # noqa: PLC0415

        # Extract host + relative path from the site URL.
        from urllib.parse import urlparse
        parsed = urlparse(self.config.site_url)
        host = parsed.netloc
        path = parsed.path.rstrip("/")
        url = f"{self._GRAPH_BASE}/sites/{host}:{path}"
        resp = requests.get(url, headers=headers, timeout=15)
        if not resp.ok:
            raise RuntimeError(f"Could not resolve SharePoint site: {resp.status_code} {resp.text[:200]}")
        return str(resp.json()["id"])

    def _list_drive_items(self, site_id: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        """List all files in the configured document library folder.

        Handles Graph API pagination automatically.

        Parameters
        ----------
        site_id:
            Graph site ID obtained from :meth:`_resolve_site_id`.
        headers:
            Authorised HTTP headers.

        Returns
        -------
        list[dict[str, Any]]
            Raw Graph API drive-item dicts (each has ``name``, ``size``,
            ``@microsoft.graph.downloadUrl`` when not folder).
        """
        import requests  # noqa: PLC0415

        # Encode the library path for a folder-children query.
        library_path = self.config.library_path.strip("/")
        url = f"{self._GRAPH_BASE}/sites/{site_id}/drive/root:/{library_path}:/children"
        items: list[dict[str, Any]] = []

        while url:
            resp = requests.get(url, headers=headers, timeout=30)
            if not resp.ok:
                raise RuntimeError(
                    f"Error listing SharePoint library: {resp.status_code} {resp.text[:200]}"
                )
            data = resp.json()
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink", "")  # follow pagination

        return items

    def _download_item(
        self,
        item: dict[str, Any],
        headers: dict[str, str],
        result: SharePointSyncResult,
    ) -> None:
        """Download a single drive item to ``downloads_dir``.

        Skips the download when a local file with the same name and byte-size
        already exists (avoids re-downloading unchanged files on every sync).

        Parameters
        ----------
        item:
            Graph API drive-item dict.
        headers:
            Authorised HTTP headers.
        result:
            :class:`SharePointSyncResult` mutated in-place to record
            download/skip outcomes.
        """
        import requests  # noqa: PLC0415

        name: str = item.get("name", "unknown")
        remote_size: int = int(item.get("size", -1))
        dest: Path = self.downloads_dir / name

        # Freshness check — skip if same-size file already present locally.
        if dest.exists() and remote_size > 0 and dest.stat().st_size == remote_size:
            result.skipped.append(name)
            return

        download_url: str = item.get("@microsoft.graph.downloadUrl", "")
        if not download_url:
            result.skipped.append(name)
            return

        resp = requests.get(download_url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)

        result.downloaded.append(dest)
