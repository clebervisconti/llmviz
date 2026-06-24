# LLMViz — Deployment runbook

> ✅ **DEPLOYED 2026-06-13** → https://llmviz.cybersphere.com.br (DEMO default; live NANO+MICRO
> real GPT-2; SMALL scripted). Repo: https://github.com/clebervisconti/llmviz. The steps below
> are the as-built record. Benchmarks: DistilGPT-2 654MB/46ms, GPT-2 634MB/452ms per forward pass
> (with attentions) on the 2-vCPU box. LLMViz cgroup peak ~770MB, capped at 2G. `MAX_RESIDENT=1`.


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

**Automated since 2026-06-24** via a Claude Code **Stop hook** (`.claude/sync-and-deploy.sh`,
registered in `.claude/settings.json`). On every Stop it auto-commits local changes, pushes to
`origin/main`, and — only when new commits land on origin — SSHes to the VPS, `git pull --ff-only`s
`/var/lib/llmviz/repo`, and restarts `llmviz`. It always exits 0 (never blocks the session),
single-flights via a lock, and logs to `.claude/deploy.log` (gitignored). SSH key comes from the
Keychain (`hostgator-vps-ssh-key`); host/port from `~/.env.hostgator` — no secrets in the repo.
To pause auto-deploy, remove the `Stop` block from `.claude/settings.json`. Manual pull still works.

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

## Observability — Splunk Observability Cloud (realm `us1`)  *(added 2026-06-15)*

Three pillars, all flowing to `app.us1.observability.splunkcloud.com`:

1. **Host infrastructure** — Splunk OTel Collector (`splunk-otel-collector` RPM, agent mode)
   ships `hostmetrics` → SignalFx. Installed manually (the `dl.signalfx.com` installer doesn't
   recognise AlmaLinux): repo `/etc/yum.repos.d/splunk-otel-collector.repo` (`repo_gpgcheck=0`,
   `gpgcheck=1`), env `/etc/otel/collector/splunk-otel-collector.conf` (`SPLUNK_REALM=us1`,
   `SPLUNK_ACCESS_TOKEN`=INGEST token, `SPLUNK_MEMORY_TOTAL_MIB=256`, `SPLUNK_LISTEN_INTERFACE=127.0.0.1`).
   Memory drop-in `…/splunk-otel-collector.service.d/10-memcap.conf` caps it at 350M (uses ~40M).
   ⚠ Internal telemetry/Prometheus moved 8888→8889 in `agent_config.yaml` (8888 is CyberPanel's
   `fastapi_ssh_server`). Backup: `agent_config.yaml.orig`.
2. **APM (traces + app metrics)** — app runs under `opentelemetry-instrument` (see `deploy/llmviz.service`),
   exports OTLP to the local collector `127.0.0.1:4317`. `service.name=llmviz`, `env=production`.
   Dep: `splunk-opentelemetry` (run `opentelemetry-bootstrap -a install` once after a venv rebuild).
3. **RUM (user experience)** — `@splunk/otel-web` snippet in `static/index.html`, `app=llmviz-browser`.
   RUM token is browser-public + RUM-scoped only; propagates trace context to same-origin `/api/*`.

**Tokens:** never in git. INGEST token lives only in the collector conf on the VPS; RUM token is
public-by-design in the HTML. Both originate from `~/.env.splunk` / macOS Keychain on Cleber's Mac
(see global `CLAUDE.md`). To rotate: update the collector conf (`systemctl restart splunk-otel-collector`)
and/or the `rumAccessToken` in `index.html` (redeploy).

**Verify flow:** `curl -s http://127.0.0.1:8889/metrics | grep otelcol_exporter_sent_` on the VPS
(metric_points + spans counters should climb, send_failed should stay absent/0).
