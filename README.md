# QU.AI

Minimal runtime repo for the local QU.AI chatbox.

Included:
- `src/apps/qshll_chatbox.py`
- local Ollama chat runtime
- RFTMW compressed-pack runtime
- RFT/topological support code required by the chatbox
- launch scripts for normal and RFTMW-first startup
- bundle staging helper for repo-local `.rftmwpk` packs
- external-pack fetch helper for large `.rftmwpk` bundles

Not included:
- proofs/workspace material
- validation/tests
- broad project docs
- unrelated apps and experiments

## Run

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

Launch the chatbox:

```powershell
.\qshellchatbox.ps1
```

Launch with RFTMW-first discovery:

```powershell
.\qshellchatbox_rftmw.ps1
```

## Portable RFTMW Bundle

To stage a repo-local compressed pack bundle:

```bash
python src/apps/stage_chatbox_rftmw_bundle.py \
  --pack <path-to-model.rftmwpk> \
  --model-dir <path-to-model-config-tokenizer-dir> \
  --set-default
```

This creates:

```text
ai/runtime/chatbox/
  manifest.json
  packs/
  models/
```

If the compressed path is active, the Session card will show:

```text
Local LLM: rftmw:... [pack:...]
```

## Qwen Pack Download

The rebuilt direct-GGUF `qwen2.5-coder:3b` pack is hosted externally because the
binary is too large for normal GitHub repo storage.

- Google Drive: `https://drive.google.com/file/d/1AHPK6lvrcb_3XhkPCsAf96Xj9uD0cbog/view?usp=sharing`
- Expected filename: `qwen2.5-coder_3b.rftmwpk`
- Size: `2,128,004,908` bytes
- Source GGUF SHA-256: `4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba`
- Pack provenance SHA-256: `7ad9899be6aa2643efb160d044253ca7866b0e3525c986a2428c2a4ce72784c3`

Fetch it into the repo-local runtime path with:

```bash
python src/apps/fetch_chatbox_pack.py
```

That places the pack at:

```text
ai/runtime/chatbox/packs/qwen2.5-coder_3b.rftmwpk
```
