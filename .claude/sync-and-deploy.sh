#!/usr/bin/env bash
#
# sync-and-deploy.sh — Claude Code **Stop** hook for LLMViz.
#
# On every Stop it:
#   1. Auto-commits any local changes (timestamped message).
#   2. Pushes to GitHub (origin/main).
#   3. If — and only if — new commits reached origin, SSHes to the HostGator VPS,
#      `git pull --ff-only`s the live repo, and restarts the systemd service.
#
# Design notes:
#   - Always exits 0: a deploy hiccup must never block the editing session.
#   - A lock file makes overlapping runs no-ops (Stop can fire in quick succession).
#   - All output is appended to .claude/deploy.log (gitignored).
#   - No secrets in this file. SSH key path comes from the macOS Keychain; host/port
#     from ~/.env.hostgator. Mirrors the credential map in the global CLAUDE.md.
#
set -uo pipefail

# --- locate repo root (this script lives in <repo>/.claude/) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR" || exit 0

LOG="$REPO_DIR/.claude/deploy.log"
LOCK="$REPO_DIR/.claude/.deploy.lock"
BRANCH="main"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }

# --- single-flight lock (no-op if another run holds it) ----------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  log "skip: another sync-and-deploy run is in progress"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# --- only operate on a real git repo on the expected branch ------------------
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { log "skip: not a git repo"; exit 0; }
cur_branch="$(git symbolic-ref --quiet --short HEAD || echo DETACHED)"
if [ "$cur_branch" != "$BRANCH" ]; then
  log "skip: on branch '$cur_branch', not '$BRANCH'"
  exit 0
fi

remote_before="$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo none)"

# --- 1. auto-commit local changes -------------------------------------------
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  msg="chore: auto-sync via Claude Code Stop hook ($(date '+%Y-%m-%d %H:%M:%S'))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  if git commit -q -m "$msg"; then
    log "committed local changes"
  else
    log "warn: git commit reported nothing to do"
  fi
else
  log "no local changes to commit"
fi

# --- 2. push (also flushes any pre-existing unpushed commits) ----------------
if git push -q origin "$BRANCH" 2>>"$LOG"; then
  remote_after="$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo none)"
else
  log "ERROR: git push failed — skipping VPS deploy (will retry next Stop)"
  exit 0
fi

if [ "$remote_before" = "$remote_after" ]; then
  log "nothing new on origin/$BRANCH — VPS already current, skipping deploy"
  exit 0
fi
log "pushed ${remote_before:0:7} -> ${remote_after:0:7}"

# --- 3. deploy to the VPS ----------------------------------------------------
# shellcheck disable=SC1090
[ -f "$HOME/.env.hostgator" ] && { set -a; . "$HOME/.env.hostgator"; set +a; }
HG_HOST="${HG_HOST:-129.121.49.96}"
HG_PORT="${HG_PORT:-22022}"
HG_USER="${HG_USER:-root}"
SSH_KEY="$(security find-generic-password -a "${USER:-$(whoami)}" -s "hostgator-vps-ssh-key" -w 2>/dev/null)"
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
  log "ERROR: SSH key not found in Keychain ('hostgator-vps-ssh-key') — code is on GitHub but VPS NOT updated"
  exit 0
fi

remote_cmd='set -e; cd /var/lib/llmviz/repo && git pull --ff-only && systemctl restart llmviz && echo OK'
if ssh -p "$HG_PORT" -i "$SSH_KEY" \
       -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
       "$HG_USER@$HG_HOST" "$remote_cmd" >>"$LOG" 2>&1; then
  log "DEPLOYED to VPS — llmviz restarted"
else
  log "ERROR: VPS deploy failed (code is on GitHub; pull/restart on the box to recover)"
fi

exit 0
