# QUAI Mobile

Expo Router shell for the hosted QU.AI runtime.

## What it expects

- A hosted runtime exposing `/api/runtime/bootstrap`
- Optional update, research, and route endpoints from `src/apps/runtime_host_api.py`
- `EXPO_PUBLIC_QUAI_RUNTIME_URL` pointing at that hosted runtime

## Local dev

```bash
npm install
npx expo start
```

Set the runtime URL before launch:

```bash
EXPO_PUBLIC_QUAI_RUNTIME_URL=http://localhost:8787
```

## Screens

- `src/app/index.tsx` for runtime health and default bundle status
- `src/app/updates.tsx` for deployment/runtime updates
- `src/app/research.tsx` for Oracle storage and hosting notes
- `src/app/routing.tsx` for client and API route maps
