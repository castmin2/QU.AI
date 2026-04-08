#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.apps.fetch_chatbox_pack import ensure_pack
from src.apps.runtime_bootstrap import (
    build_effective_bundle,
    load_chatbox_manifest,
    runtime_bootstrap_snapshot,
)


WORKSPACE_NOTES_PATH = PROJECT_ROOT / "dev" / "runtime_workspace.json"
SERVER_STATE: dict[str, Any] = {
    "prefetch_error": "",
    "prefetch_path": "",
    "started_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}


def _load_workspace_notes() -> dict[str, Any]:
    if not WORKSPACE_NOTES_PATH.exists():
        return {
            "generated_at_utc": SERVER_STATE["started_at_utc"],
            "updates": [],
            "research": [],
            "routes": [],
            "api_routes": [],
        }
    return json.loads(WORKSPACE_NOTES_PATH.read_text(encoding="utf-8"))


def _prefetch_default_pack(force: bool = False) -> None:
    try:
        pack_path = ensure_pack("", force=force)
        SERVER_STATE["prefetch_path"] = str(pack_path)
        SERVER_STATE["prefetch_error"] = ""
    except Exception as exc:
        SERVER_STATE["prefetch_error"] = f"{type(exc).__name__}: {exc}"


class RuntimeHandler(BaseHTTPRequestHandler):
    server_version = "QUAIRuntimeHost/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[runtime-host] {self.address_string()} - {fmt % args}")

    def _write_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._write_json({"ok": True}, status=HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        notes = _load_workspace_notes()

        if parsed.path == "/":
            self._write_json(
                {
                    "service": "quai-runtime-host",
                    "started_at_utc": SERVER_STATE["started_at_utc"],
                    "endpoints": [
                        "/healthz",
                        "/api/runtime/bootstrap",
                        "/api/runtime/manifest",
                        "/api/runtime/updates",
                        "/api/runtime/research",
                        "/api/runtime/routes",
                    ],
                }
            )
            return

        if parsed.path == "/healthz":
            snapshot = runtime_bootstrap_snapshot()
            pack_status = snapshot.get("pack_status") if isinstance(snapshot.get("pack_status"), dict) else {}
            degraded = bool(SERVER_STATE["prefetch_error"])
            self._write_json(
                {
                    "ok": not degraded,
                    "prefetch_error": SERVER_STATE["prefetch_error"],
                    "prefetch_path": SERVER_STATE["prefetch_path"],
                    "pack_ready": bool(pack_status.get("exists")) and bool(pack_status.get("size_matches")),
                },
                status=HTTPStatus.OK if not degraded else HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        if parsed.path == "/api/runtime/bootstrap":
            snapshot = runtime_bootstrap_snapshot()
            snapshot["workspace"] = {
                "updates_count": len(notes.get("updates", [])),
                "research_count": len(notes.get("research", [])),
                "routes_count": len(notes.get("routes", [])),
            }
            snapshot["prefetch"] = {
                "error": SERVER_STATE["prefetch_error"],
                "path": SERVER_STATE["prefetch_path"],
            }
            self._write_json(snapshot)
            return

        if parsed.path == "/api/runtime/manifest":
            manifest = load_chatbox_manifest()
            _, bundle = build_effective_bundle()
            self._write_json({"manifest": manifest, "effective_bundle": bundle})
            return

        if parsed.path == "/api/runtime/updates":
            self._write_json({"items": notes.get("updates", [])})
            return

        if parsed.path == "/api/runtime/research":
            self._write_json({"items": notes.get("research", [])})
            return

        if parsed.path == "/api/runtime/routes":
            self._write_json(
                {
                    "screens": notes.get("routes", []),
                    "api_routes": notes.get("api_routes", []),
                }
            )
            return

        self._write_json({"error": "not_found", "path": parsed.path}, status=HTTPStatus.NOT_FOUND)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expose QU.AI runtime bootstrap data for Replit and Expo clients.")
    ap.add_argument("--host", default=os.getenv("QUAI_RUNTIME_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("QUAI_RUNTIME_PORT", "8787")))
    ap.add_argument("--prefetch-default-pack", action="store_true")
    args = ap.parse_args()

    if args.prefetch_default_pack or os.getenv("QUAI_PREFETCH_DEFAULT_PACK", "0") == "1":
        _prefetch_default_pack(force=False)

    server = ThreadingHTTPServer((args.host, args.port), RuntimeHandler)
    print(f"QU.AI runtime host listening on http://{args.host}:{args.port}")
    if SERVER_STATE["prefetch_error"]:
        print(f"Prefetch warning: {SERVER_STATE['prefetch_error']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(f"{type(exc).__name__}: {exc}")
