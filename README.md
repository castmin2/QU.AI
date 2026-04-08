# QU.AI

Minimal runtime repo for the local QU.AI chatbox, hosted runtime bootstrap, and the first Expo mobile shell.

Included:
- `src/apps/qshll_chatbox.py`
- local Ollama chat runtime
- RFTMW compressed-pack runtime
- repo-local chatbox bundle manifest and metadata
- hosted pack fetch/bootstrap helpers
- lightweight Replit-friendly runtime host API
- Expo Router frontend scaffold in `frontend/expo-ai`

Not included:
- proofs/workspace material
- validation/tests
- unrelated apps and experiments

## Local Runtime

Install the Python runtime dependencies:

```bash
pip install -r requirements.txt
```

Launch the desktop chatbox:

```powershell
.\qshellchatbox.ps1
```

Launch with RFTMW-first discovery:

```powershell
.\qshellchatbox_rftmw.ps1
```

## Default Pack

The default bundle is the rebuilt direct-GGUF `qwen2.5-coder:3b` pack.

- Filename: `qwen2.5-coder_3b.rftmwpk`
- Size: `2,128,004,908` bytes
- Pack file SHA-256: `502112d74e76d8d1d38c6189a038cc45ffd229fb773f557067dc21c3355e0d33`
- Source GGUF SHA-256: `4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba`
- Pack provenance SHA-256: `7ad9899be6aa2643efb160d044253ca7866b0e3525c986a2428c2a4ce72784c3`
- Google Drive fallback: `https://drive.google.com/file/d/1AHPK6lvrcb_3XhkPCsAf96Xj9uD0cbog/view?usp=sharing`

Fetch the pack into the repo-local runtime path with:

```bash
python src/apps/fetch_chatbox_pack.py
```

## Oracle Bucket / Replit

Hosted environments can override the pack source at runtime without editing `manifest.json`.

Preferred env vars:
- `ORACLE_OBJECT_PREAUTH_URL`
- `QUAI_PACK_FILE_SHA256`
- `QUAI_PACK_SIZE_BYTES`
- `QUAI_PREFETCH_DEFAULT_PACK=1`

Start the hosted runtime contract locally:

```bash
python src/apps/runtime_host_api.py --prefetch-default-pack
```

That exposes:

- `/healthz`
- `/api/runtime/bootstrap`
- `/api/runtime/manifest`
- `/api/runtime/updates`
- `/api/runtime/research`
- `/api/runtime/routes`

Replit scaffold files are included:

- `.replit`
- `replit.nix`
- `replit.md`
- `.env.example`

## Expo Frontend

The mobile shell lives in `frontend/expo-ai` and uses Expo Router.

Install and run it with:

```bash
cd frontend/expo-ai
npm install
npx expo start
```

Set the hosted runtime URL for the mobile client with:

```bash
EXPO_PUBLIC_QUAI_RUNTIME_URL=http://localhost:8787
```
