#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.apps.runtime_bootstrap import build_effective_bundle, bundle_target_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_html(blob: bytes) -> bool:
    head = blob[:256].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _parse_google_drive_confirm(html_text: str, current_url: str) -> str:
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html_text, re.IGNORECASE)
    if not action_match:
        raise RuntimeError("Google Drive returned an HTML page without a downloadable form.")

    action = html.unescape(action_match.group(1))
    params = {
        key: html.unescape(value)
        for key, value in re.findall(
            r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
            html_text,
            re.IGNORECASE,
        )
    }
    if not params:
        raise RuntimeError("Google Drive confirmation page did not include download parameters.")

    action_url = urllib.parse.urljoin(current_url, action)
    return f"{action_url}?{urllib.parse.urlencode(params)}"


def _stream_http_download(url: str, target_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 QU.AI pack fetcher"})
    with urllib.request.urlopen(req, timeout=3600) as resp, target_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _download_with_google_drive_support(url: str, target_path: Path) -> None:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    headers = {"User-Agent": "Mozilla/5.0 QU.AI pack fetcher"}

    def _open(url_to_open: str):
        req = urllib.request.Request(url_to_open, headers=headers)
        return opener.open(req, timeout=3600)

    with _open(url) as resp:
        first_chunk = resp.read(8192)
        if _looks_like_html(first_chunk):
            html_body = first_chunk + resp.read()
            confirm_url = _parse_google_drive_confirm(html_body.decode("utf-8", errors="replace"), resp.geturl())
        else:
            with target_path.open("wb") as f:
                f.write(first_chunk)
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return

    with _open(confirm_url) as resp:
        first_chunk = resp.read(8192)
        if _looks_like_html(first_chunk):
            html_body = first_chunk + resp.read()
            raise RuntimeError(
                "Google Drive still returned HTML instead of the pack binary. "
                f"Response started with: {html_body[:160].decode('utf-8', errors='replace')}"
            )
        with target_path.open("wb") as f:
            f.write(first_chunk)
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def _download_pack(download: dict, target_path: Path) -> None:
    provider = str(download.get("provider") or "").strip().lower()
    direct_url = str(download.get("direct_url") or "").strip()
    if not direct_url:
        raise SystemExit("No external pack download URL is recorded for this bundle.")

    if provider == "google_drive" or "drive.google.com" in direct_url:
        _download_with_google_drive_support(direct_url, target_path)
        return

    _stream_http_download(direct_url, target_path)


def ensure_pack(bundle_id: str = "", *, force: bool = False) -> Path:
    _, bundle = build_effective_bundle(bundle_id)
    download = bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {}
    expected_sha = str(download.get("pack_file_sha256") or "").strip().lower()
    expected_size = int(download.get("size_bytes", 0) or 0)
    target_path = bundle_target_path(bundle)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        current_size = target_path.stat().st_size
        current_sha = _sha256(target_path) if expected_sha else ""
        if (not expected_size or current_size == expected_size) and (not expected_sha or current_sha == expected_sha):
            print(f"Pack already exists: {target_path}")
            if expected_sha:
                print(f"Current SHA-256: {current_sha}")
            return target_path
        print("Existing local pack is invalid or incomplete; re-downloading.")
        if not force:
            target_path.unlink(missing_ok=True)

    temp_path = target_path.with_suffix(target_path.suffix + ".partial")
    temp_path.unlink(missing_ok=True)

    try:
        _download_pack(download, temp_path)
        if expected_size:
            actual_size = temp_path.stat().st_size
            print(f"Size bytes: {actual_size}")
            if actual_size != expected_size:
                raise SystemExit(
                    "Downloaded pack size does not match manifest size. "
                    f"expected={expected_size} actual={actual_size}"
                )

        if expected_sha:
            actual_sha = _sha256(temp_path).lower()
            print(f"SHA-256: {actual_sha}")
            if actual_sha != expected_sha:
                raise SystemExit(
                    "Downloaded pack hash does not match manifest file hash. "
                    f"expected={expected_sha} actual={actual_sha}"
                )

        os.replace(temp_path, target_path)
        print("Pack ready.")
        return target_path
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch an externally hosted chatbox .rftmwpk into ai/runtime/chatbox/packs/")
    ap.add_argument("--bundle-id", default="", help="Bundle id from ai/runtime/chatbox/manifest.json")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing local pack")
    args = ap.parse_args()

    _, bundle = build_effective_bundle(args.bundle_id)
    download = bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {}
    target_path = bundle_target_path(bundle)

    print(f"Downloading: {download.get('direct_url', '')}")
    print(f"Target: {target_path}")
    ensure_pack(args.bundle_id, force=args.force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
