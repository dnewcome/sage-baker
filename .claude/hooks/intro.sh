#!/usr/bin/env bash
# SessionStart banner: list available skills + key commands.
# Skills are read from .claude/skills/*/SKILL.md so this stays current.

set -euo pipefail

cat >&2 <<'BANNER'
─────────────────────────────────────────────────────────────
 sage-baker — personal ML pipeline sandbox
─────────────────────────────────────────────────────────────

Skills (slash-commands):
BANNER

shopt -s nullglob
found=0
for f in .claude/skills/*/SKILL.md; do
    name=$(basename "$(dirname "$f")")
    desc=$(awk -F': ' '/^description:/{ $1=""; sub(/^ /,""); print; exit }' "$f")
    # Trim at first em-dash or period (whichever first), cap at 70 chars.
    short=$(printf '%s' "$desc" | sed -E 's/[—.].*$//' | cut -c1-70)
    printf "  /%-18s %s\n" "$name" "$short" >&2
    found=1
done
[ "$found" = "1" ] || echo "  (none yet — drop one in .claude/skills/<name>/SKILL.md)" >&2

cat >&2 <<'BANNER'

Common commands:
  make help          full list of make targets
  make data-sonar    fetch sonar dataset
  make train         train default plugin (sonar)
  make agent         autoresearch loop on default plugin
  make jupyter       launch JupyterLab with .env exported

Reference: README.md (full docs), PLAN.md (phase status)
─────────────────────────────────────────────────────────────
BANNER
