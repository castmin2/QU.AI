# QU.AI Replit Workspace

## Project Context
- This repo hosts the lightweight QU.AI runtime, pack manifest, and mobile-facing bootstrap endpoints.
- The default local bundle is `qwen2.5-coder:3b` backed by `qwen2.5-coder_3b.rftmwpk`.
- Large `.rftmwpk` files stay outside git and are fetched into `ai/runtime/chatbox/packs/`.

## Current Development Phase
- Keep the hosted runtime thin: expose health, manifest, updates, research, and routing endpoints.
- Use Oracle Object Storage pre-authenticated URLs as the preferred hosted pack prerequisite path.
- Keep the Expo frontend and hosted runtime on one clear contract: `/api/runtime/bootstrap`.

## Runtime Rules
- Prefer stdlib Python for hosted bootstrap services.
- Do not add heavyweight backend frameworks unless the app outgrows the simple host API.
- Treat `ORACLE_OBJECT_PREAUTH_URL` as the canonical override for pack delivery in hosted environments.
- Preserve the existing local Google Drive fallback in `manifest.json`.

## Frontend Rules
- Frontend lives in `frontend/expo-ai`.
- Use Expo Router with file-based routing.
- Keep route names aligned with the workspace notes: `index`, `updates`, `research`, `routing`.
- Mobile/web clients should read `EXPO_PUBLIC_QUAI_RUNTIME_URL` and consume the runtime host endpoints instead of hardcoding pack details.

## Environment
- Replit secrets should hold hosted URLs and checksums.
- Supported pack override env vars:
  - `ORACLE_OBJECT_PREAUTH_URL`
  - `QUAI_PACK_DIRECT_URL`
  - `QUAI_PACK_FILE_SHA256`
  - `QUAI_PACK_SIZE_BYTES`
  - `QUAI_PREFETCH_DEFAULT_PACK`

## Useful Commands
- Hosted runtime: `python src/apps/runtime_host_api.py --prefetch-default-pack`
- Fetch bundle only: `python src/apps/fetch_chatbox_pack.py`
- Launch local desktop chatbox: `./qshellchatbox_rftmw.ps1`
- Expo frontend: `cd frontend/expo-ai && npm install && npx expo start`
