# LLMViz on the Mac mini (migrated off the VPS, 2026-06-28)

LLMViz now runs on the **Mac mini** and is served publicly via the existing **agentos
cloudflared tunnel**. This unlocks the full feature set the VPS couldn't: real white-box
Q/K/V + attention (GPT-2 family on `torch`/MPS) **and** Gemma-2-9B as a black-box outputs tier.

## What's wired

- **App**: FastAPI on `127.0.0.1:8810`, env `LLMVIZ_LIVE=1`,
  `LLMVIZ_LIVE_TIERS=nano,micro,small`, `LLMVIZ_MLX_URL=http://localhost:8081`.
  Tiers: DEMO (scripted) · NANO distilgpt2 · MICRO gpt2 · SMALL gpt2-medium (all real,
  white-box) · GEMMA gemma-2-9b-it-4bit via MLX (real tokens+probabilities, **no** attention).
- **Gemma/MLX**: `mlx-watchdog.py` runs `mlx_lm.server` on `127.0.0.1:8081` serving
  **`mlx-community/gemma-3-4b-it-4bit`** (Gemma 3 4B), model cached at
  `/Volumes/STORAGE/Cyberlabs/models`. **Runs from a dedicated Python 3.11 venv
  (`/Volumes/STORAGE/Cyberlabs/mlx-venv`, `mlx_lm 0.31.3` / `mlx 0.31.2`)** — the system
  Python 3.9 tops out at `mlx 0.29.x`, which has a Gemma 3 sliding-window KV-cache bug
  (`RotatingKVCache._temporal_order` broadcast error). mlx_lm ≥ 0.30 also switched to the
  OpenAI logprobs shape (`logprobs:true` + `top_logprobs:N`); `app/mlx_backend.py` handles both.
  Start it: `MLX_MODEL=mlx-community/gemma-3-4b-it-4bit MLX_PORT=8081
  HF_HUB_CACHE=/Volumes/STORAGE/Cyberlabs/models /Volumes/STORAGE/Cyberlabs/mlx-venv/bin/python
  /Volumes/STORAGE/Cyberlabs/mlx-watchdog.py`
- **Tunnel**: `~/.cloudflared/config.yml` ingress `llmviz.cybersphere.com.br → localhost:8810`
  on the `agentos` tunnel (`a2f53637…`). Config backed up to `config.yml.bak.llmviz.*`.
- **DNS**: `llmviz.cybersphere.com.br` is a proxied CNAME → `a2f53637….cfargotunnel.com`
  (was an A record → the VPS `129.121.49.96`). No Cloudflare Access gate (public).

## Run it now (from a Terminal — has disk access)

```bash
cd "/Volumes/STORAGE/Cyberlabs/Web Apps/LLMViz" && ./run-local.sh   # NANO+MICRO
# or: LLMVIZ_LIVE_TIERS=nano,micro,small ./run-local.sh             # + gpt2-medium
```
Gemma: `HF_HUB_CACHE=/Volumes/STORAGE/Cyberlabs/models /usr/bin/python3 /Volumes/STORAGE/Cyberlabs/mlx-watchdog.py`

## ⚠️ Boot-persistence requires a one-time Full Disk Access grant

`launchd` agents are denied TCC access to `/Volumes/STORAGE/Cyberlabs`, so Python launched by
launchd fails (`Operation not permitted: .venv/pyvenv.cfg`; SIP also strips `PYTHONPATH`). This
is the same reason the MLX server wasn't surviving reboots. cloudflared works under launchd only
because its binary was previously granted disk access. **Both LLMViz and MLX need the Python
interpreter granted Full Disk Access** to auto-start at boot.

**Grant it (System Settings — only you can do this; Claude is not allowed to change security settings):**
1. System Settings → Privacy & Security → **Full Disk Access**.
2. Click **+**. In the file picker press **⌘⇧G** and paste:
   `/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python`
   (this binary backs both the venv python and `/usr/bin/python3`). Add it and toggle it **on**.
3. (Optional, belt-and-suspenders) also add `/bin/bash`.

**Then the launchd agents (already installed) take over:**
```bash
# LLMViz — plist installed at /Users/clebervisconti/Library/LaunchAgents/com.cyberlabs.llmviz.plist
launchctl kickstart -k gui/$(id -u)/com.cyberlabs.llmviz
# MLX/Gemma — existing agent
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cyberlabs.mlx-server.plist 2>/dev/null
launchctl kickstart -k gui/$(id -u)/com.cyberlabs.mlx-server
```
The `com.cyberlabs.llmviz` plist uses the AgentOS pattern (internal-disk install,
`wait-for-storage-llmviz.sh` boot-race guard, venv python via `--app-dir`). Until FDA is granted
it fails with the TCC error above; after, it serves on every boot.

## Restart after a code change
```bash
launchctl kickstart -k gui/$(id -u)/com.cyberlabs.llmviz   # if launchd (post-FDA)
# or just re-run ./run-local.sh from a Terminal
```
