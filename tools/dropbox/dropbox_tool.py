#!/usr/bin/env python3
"""Upload and download project artifacts with Dropbox.

The tool uses Dropbox's refresh-token flow, verifies Dropbox content hashes,
and keeps project artifacts below /rsrch_new by default. It depends only on the
Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATHS = (PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env")
DEFAULT_DROPBOX_ROOT = "/rsrch_new"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "dropbox"
CONTENT_BLOCK_SIZE = 4 * 1024 * 1024
UPLOAD_SESSION_THRESHOLD = 100 * 1024 * 1024


class DropboxError(RuntimeError):
    """A Dropbox request failed or returned invalid data."""


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            parsed = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise DropboxError(f"invalid environment value in {path} on line {line_number}") from exc
        os.environ.setdefault(key, parsed[0] if parsed else "")


def _load_project_env() -> None:
    for path in ENV_PATHS:
        _load_env(path)


def _dropbox_root() -> str:
    configured = os.environ.get("DROPBOX_PROJECT_FOLDER", DEFAULT_DROPBOX_ROOT).strip()
    if not configured or configured == "/":
        raise DropboxError("DROPBOX_PROJECT_FOLDER must name a project folder, not Dropbox root")
    return "/" + configured.strip("/")


def _access_token() -> str:
    _load_project_env()
    static_token = os.environ.get("DROPBOX_ACCESS_TOKEN", "").strip()
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN", "").strip()
    app_key = os.environ.get("DROPBOX_APP_KEY", "").strip()
    app_secret = os.environ.get("DROPBOX_APP_SECRET", "").strip()

    if refresh_token:
        missing = [
            name
            for name, value in (("DROPBOX_APP_KEY", app_key), ("DROPBOX_APP_SECRET", app_secret))
            if not value
        ]
        if missing:
            raise DropboxError(f"missing Dropbox credentials: {', '.join(missing)}")
        payload = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": app_key,
                "client_secret": app_secret,
            }
        ).encode("ascii")
        response = _json_request(
            "https://api.dropboxapi.com/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=payload,
        )
        token = response.get("access_token")
        if not isinstance(token, str) or not token:
            raise DropboxError("Dropbox token response did not contain an access token")
        return token

    if static_token:
        return static_token
    raise DropboxError(
        "set DROPBOX_REFRESH_TOKEN with DROPBOX_APP_KEY and DROPBOX_APP_SECRET "
        "in .env (or set DROPBOX_ACCESS_TOKEN)"
    )


def _raw_request(
    url: str,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    data: bytes = b"",
) -> tuple[bytes, dict[str, str]]:
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DropboxError(f"Dropbox API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DropboxError(f"Dropbox request failed: {exc.reason}") from exc


def _json_request(
    url: str,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    data: bytes = b"",
    allow_null: bool = False,
) -> dict[str, Any]:
    body, _ = _raw_request(url, token=token, headers=headers, data=data)
    if not body:
        return {}
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DropboxError("Dropbox returned an invalid JSON response") from exc
    if result is None and allow_null:
        return {}
    if not isinstance(result, dict):
        raise DropboxError("Dropbox returned an unexpected JSON response")
    return result


def _rpc(token: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _json_request(
        f"https://api.dropboxapi.com/2/{method}",
        token=token,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )


def _content_request(token: str, method: str, args: dict[str, Any], data: bytes) -> dict[str, Any]:
    return _json_request(
        f"https://content.dropboxapi.com/2/{method}",
        token=token,
        headers={
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps(args, separators=(",", ":")),
        },
        data=data,
        # append_v2 returns JSON null on success, rather than file metadata.
        allow_null=method == "files/upload_session/append_v2",
    )


def _ensure_folder(token: str, remote_path: str) -> None:
    try:
        _rpc(token, "files/create_folder_v2", {"path": remote_path, "autorename": False})
    except DropboxError as exc:
        if "conflict" not in str(exc) or "folder" not in str(exc):
            raise


def _content_hash(blocks: Iterable[bytes]) -> str:
    overall = hashlib.sha256()
    for block in blocks:
        overall.update(hashlib.sha256(block).digest())
    return overall.hexdigest()


def _file_content_hash(path: Path) -> str:
    def blocks() -> Iterable[bytes]:
        with path.open("rb") as handle:
            while block := handle.read(CONTENT_BLOCK_SIZE):
                yield block

    return _content_hash(blocks())


def _bytes_content_hash(data: bytes) -> str:
    return _content_hash(
        data[offset : offset + CONTENT_BLOCK_SIZE]
        for offset in range(0, len(data), CONTENT_BLOCK_SIZE)
    )


def _upload(token: str, source: Path, destination: str) -> dict[str, Any]:
    commit = {"path": destination, "mode": "overwrite", "autorename": False, "mute": False}
    if source.stat().st_size <= UPLOAD_SESSION_THRESHOLD:
        return _content_request(token, "files/upload", commit, source.read_bytes())

    with source.open("rb") as handle:
        first_chunk = handle.read(CONTENT_BLOCK_SIZE)
        started = _content_request(token, "files/upload_session/start", {"close": False}, first_chunk)
        session_id = started.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise DropboxError("Dropbox did not return an upload session ID")
        offset = len(first_chunk)
        while True:
            chunk = handle.read(CONTENT_BLOCK_SIZE)
            cursor = {"session_id": session_id, "offset": offset}
            if len(chunk) < CONTENT_BLOCK_SIZE:
                return _content_request(
                    token,
                    "files/upload_session/finish",
                    {"cursor": cursor, "commit": commit},
                    chunk,
                )
            _content_request(token, "files/upload_session/append_v2", {"cursor": cursor, "close": False}, chunk)
            offset += len(chunk)


def _temporary_download_link(token: str, destination: str) -> str:
    result = _rpc(token, "files/get_temporary_link", {"path": destination})
    link = result.get("link")
    if not isinstance(link, str) or not link:
        raise DropboxError(f"Dropbox did not return a temporary link for {destination}")
    return link


def upload_files(paths: list[Path], *, date_folder: str | None = None) -> list[dict[str, Any]]:
    sources = [path.expanduser().resolve() for path in paths]
    invalid = [str(path) for path in sources if not path.is_file()]
    if invalid:
        raise DropboxError(f"not a regular file: {', '.join(invalid)}")

    token = _access_token()
    date_folder = date_folder or datetime.now().astimezone().date().isoformat()
    remote_root = _dropbox_root()
    remote_folder = f"{remote_root}/{date_folder}"
    _ensure_folder(token, remote_root)
    _ensure_folder(token, remote_folder)

    results: list[dict[str, Any]] = []
    for source in sources:
        destination = f"{remote_folder}/{source.name}"
        metadata = _upload(token, source, destination)
        if metadata.get("content_hash") != _file_content_hash(source):
            raise DropboxError(f"content-hash mismatch after uploading {source}")
        link = _temporary_download_link(token, destination)
        result = {
            "source": str(source),
            "path": destination,
            "size": metadata.get("size"),
            "verified": True,
            "temporary_download_link": link,
        }
        results.append(result)
        print(f"Uploaded {source} -> {destination} ({metadata.get('size')} bytes, integrity OK)")
        print(f"Temporary direct-download link (~4h): {link}")
    return results


def _normalize_remote_path(remote_path: str | None) -> str:
    if not remote_path or not remote_path.strip():
        return _dropbox_root()
    stripped = remote_path.strip()
    if stripped.startswith("/"):
        return "/" + stripped.strip("/")
    return f"{_dropbox_root()}/{stripped.strip('/')}"


def _list_folder(token: str, folder: str) -> list[dict[str, Any]]:
    page = _rpc(
        token,
        "files/list_folder",
        {
            "path": folder,
            "recursive": True,
            "include_deleted": False,
            "include_non_downloadable_files": False,
        },
    )
    entries: list[dict[str, Any]] = []
    while True:
        page_entries = page.get("entries", [])
        if not isinstance(page_entries, list):
            raise DropboxError("Dropbox folder listing did not contain an entries list")
        entries.extend(entry for entry in page_entries if isinstance(entry, dict))
        if not page.get("has_more"):
            return entries
        cursor = page.get("cursor")
        if not isinstance(cursor, str) or not cursor:
            raise DropboxError("Dropbox folder listing is missing a continuation cursor")
        page = _rpc(token, "files/list_folder/continue", {"cursor": cursor})


def _modified_timestamp(entry: dict[str, Any]) -> float:
    value = entry.get("server_modified") or entry.get("client_modified")
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _latest_file(token: str, folder: str) -> dict[str, Any]:
    files = [entry for entry in _list_folder(token, folder) if entry.get(".tag") == "file"]
    if not files:
        raise DropboxError(f"no downloadable files found in Dropbox folder {folder}")
    return max(files, key=_modified_timestamp)


def _resolve_download_entry(token: str, remote_path: str | None) -> dict[str, Any]:
    resolved = _normalize_remote_path(remote_path)
    if remote_path is None:
        return _latest_file(token, resolved)
    metadata = _rpc(token, "files/get_metadata", {"path": resolved, "include_deleted": False})
    if metadata.get(".tag") == "file":
        return metadata
    if metadata.get(".tag") == "folder":
        return _latest_file(token, resolved)
    raise DropboxError(f"Dropbox path is not a downloadable file or folder: {resolved}")


def download_file(
    remote_path: str | None = None,
    *,
    local_dir: Path = DEFAULT_DOWNLOAD_DIR,
    output: Path | None = None,
) -> dict[str, Any]:
    token = _access_token()
    entry = _resolve_download_entry(token, remote_path)
    source_path = entry.get("path_display") or entry.get("path_lower")
    name = entry.get("name")
    if not isinstance(source_path, str) or not source_path or not isinstance(name, str) or not name:
        raise DropboxError("Dropbox file metadata is missing its path or name")

    body, headers = _raw_request(
        "https://content.dropboxapi.com/2/files/download",
        token=token,
        headers={"Dropbox-API-Arg": json.dumps({"path": source_path}, separators=(",", ":"))},
    )
    header_metadata: dict[str, Any] = {}
    metadata_text = next((value for key, value in headers.items() if key.lower() == "dropbox-api-result"), "")
    if metadata_text:
        try:
            parsed = json.loads(metadata_text)
            if isinstance(parsed, dict):
                header_metadata = parsed
        except json.JSONDecodeError:
            pass
    expected_hash = header_metadata.get("content_hash") or entry.get("content_hash")
    actual_hash = _bytes_content_hash(body)
    if expected_hash and expected_hash != actual_hash:
        raise DropboxError(f"content-hash mismatch after downloading {source_path}")

    destination = output.expanduser().resolve() if output else (local_dir.expanduser().resolve() / name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    result = {
        "source": source_path,
        "destination": str(destination),
        "size": len(body),
        "verified": bool(expected_hash),
    }
    verification = "integrity OK" if expected_hash else "downloaded; no content hash returned"
    print(f"Downloaded {source_path} -> {destination} ({len(body)} bytes, {verification})")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload and download files for the rsrch_new Dropbox folder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="upload one or more local files")
    upload_parser.add_argument("files", nargs="+", type=Path)
    upload_parser.add_argument("--date", help="destination date folder in YYYY-MM-DD form (default: local date)")

    download_parser = subparsers.add_parser(
        "download",
        help="download a path, or the latest file under the project folder when omitted",
    )
    download_parser.add_argument(
        "remote_path",
        nargs="?",
        help="Dropbox file/folder path; relative paths are resolved below the project folder",
    )
    download_parser.add_argument("--local-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    download_parser.add_argument("--output", type=Path, help="exact local output path (files only)")

    subparsers.add_parser("check-auth", help="validate Dropbox credentials without transferring files")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "upload":
            if args.date:
                datetime.strptime(args.date, "%Y-%m-%d")
            upload_files(args.files, date_folder=args.date)
        elif args.command == "download":
            download_file(args.remote_path, local_dir=args.local_dir, output=args.output)
        elif args.command == "check-auth":
            account = _rpc(_access_token(), "users/get_current_account", {})
            display_name = account.get("name", {}).get("display_name") if isinstance(account.get("name"), dict) else None
            print(f"Dropbox credentials are valid{f' for {display_name}' if display_name else ''}.")
    except (DropboxError, ValueError) as exc:
        print(f"Dropbox {args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
