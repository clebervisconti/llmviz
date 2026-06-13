# LLMViz — Deployment runbook

> Pattern: **Shape B — dynamic app behind OLS reverse proxy** (same as NeuraNetViz, same
> playbook as AgentOS `DEPLOYMENT.md`). Skills that own each step: `cs-app-deployment`
> (pipeline), `hostgator-vps-manager` (server), `cloudflare-manager` (DNS/edge).
> Treat the VPS as production. Confirm destructive ops. Never commit secrets.

## Target topology

```
Browser ──► Cloudflare (proxied DNS, SSL) ──► VPS 129.121.49.96 (OpenLiteSpeed :443)
                                                  │  reverse proxy
                                                  ▼
                                          uvicorn 127.0.0.1:8802  (systemd: llmviz.service)
                                          /var/lib/llmviz/repo  (git clone of GitHub repo)
```

- **Subdomain**: `llmviz.cybersphere.com.br` (zone `cybersphere.com.br`,
  id `cbb665dcd1447268f6fe8b0437d1fcc6`; in `~/.env.cloudflare`).
  ⚠ This zone also hosts AgentOS behind **Cloudflare Access** — LLMViz must stay **public/no
  login**; do NOT attach an Access policy to this hostname.
- **Port**: `8802` (NeuraNetViz already owns 8801 — verify free with `ss -tlnp | grep 8802`).
- **GitHub**: public repo `clebervisconti/llmviz` (create with `gh repo create llmviz --public`).

## Steps (executed by Opus at Phase 4 — not before)

1. **GitHub repo**: `gh repo create llmviz --public --source . --push` from the project root.
2. **DNS**: via `cloudflare-manager` — A record `llmviz` in zone `cybersphere.com.br` →
   `129.121.49.96`, proxied (orange cloud), using `CF_API_TOKEN_ACCESS` from `~/.env.cloudflare`.
   Do **not** add a Cloudflare Access policy (this app is public).
3. **VPS setup** (SSH port 22022, key from Keychain `hostgator-vps-ssh-key`):
   ```bash
   mkdir -p /var/lib/llmviz && cd /var/lib/llmviz
   git clone https://github.com/clebervisconti/llmviz.git repo
   python3 -m venv venv && venv/bin/pip install -r repo/requirements.txt
   ```
   Note: install **CPU-only torch** (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
   to keep the venv small; the VPS has no GPU.
4. **systemd**: copy `deploy/llmviz.service` to `/etc/systemd/system/`, port 8802,
   `systemctl enable --now llmviz`. (Template: NeuraNetViz's `deploy/neuralnetviz.service`.)
5. **OLS reverse proxy**: CyberPanel website for the subdomain + Let's Encrypt cert
   (DNS-01, not HTTP-01 — Cloudflare proxy breaks HTTP-01), proxy context `/` →
   `http://127.0.0.1:8802`. Same config as the neuralnetviz vhost — copy and re-port it.
6. **Firewall**: no new public ports (8802 stays loopback-only).
7. **Memory guard**: in the systemd unit set `MemoryMax=2G` so a runaway inference can't
   take down the box (4GB total RAM, shared with WordPress + other apps).

## Update pipeline

Local edit → `git push` → VPS `cd /var/lib/llmviz/repo && git pull` → `systemctl restart llmviz`.
Optionally wire the AgentOS-style cron pull or a Claude Code Stop hook later; manual pull is
fine for v1.

## Post-deploy checklist

- [ ] `curl -s https://llmviz.clebervisconti.com/api/health` → `{"status":"ok", ...}`
- [ ] First generate request from the browser completes < 10s on a short prompt
- [ ] `systemctl status llmviz` clean; `journalctl -u llmviz -n 50` no tracebacks
- [ ] RAM headroom: `free -m` after one generation — process RSS within MemoryMax
- [ ] Cloudflare cache rule: no-cache on `/` and `/api/*`, short TTL (60s) on `/static/*`
- [ ] Add the app to the project registry in AgentOS `DEPLOYMENT.md` §8

## Rollback

`cd /var/lib/llmviz/repo && git reset --hard <last-good-sha> && systemctl restart llmviz`.
Before any risky server change: snapshot to `/home/backups/llmviz/$(date +%Y%m%d-%H%M%S)/`.
