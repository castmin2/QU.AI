# QU.AI

Minimal runtime repo for the local QU.AI chatbox.

Included:
- `src/apps/qshll_chatbox.py`
- local Ollama chat runtime
- RFTMW compressed-pack runtime
- RFT/topological support code required by the chatbox
- launch scripts for normal and RFTMW-first startup
- bundle staging helper for repo-local `.rftmwpk` packs

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
