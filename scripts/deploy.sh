#!/usr/bin/env bash
#
# Deploy the dashboard to GitHub and Vercel.
#
# The two logins are browser flows and cannot be automated, so this script does
# everything either side of them and stops with clear instructions if you are not
# signed in. Run it again after logging in; it is safe to re-run.
#
#   ./scripts/deploy.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
say() { printf '\n\033[1;35m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m !\033[0m %s\n' "$1"; }

# --- 1. refuse to ship secrets ----------------------------------------------
say "Checking nothing secret is staged"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "REFUSING: .env is tracked by git. Remove it: git rm --cached .env"
  exit 1
fi
if git grep -nIE 'PK[A-Z0-9]{16,}|AK[A-Z0-9]{16,}|ghp_[A-Za-z0-9]{20,}' -- \
   ':!scripts/deploy.sh' ':!docs' >/dev/null 2>&1; then
  echo "REFUSING: something key-shaped is committed. Check the match above."
  exit 1
fi
echo "    clean"

# --- 2. the dashboard has to build ------------------------------------------
say "Building the dashboard"
(cd dashboard && npm install --silent && npm run build >/dev/null)
echo "    dashboard/out ready"

# --- 3. GitHub ---------------------------------------------------------------
say "Pushing to GitHub"
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "Update dashboard snapshot and build" || true
fi

if GIT_TERMINAL_PROMPT=0 git push -u origin main 2>/dev/null; then
  echo "    pushed"
else
  warn "GitHub is not authenticated. Do one of these, then re-run this script:"
  cat <<'HELP'

      brew install gh && gh auth login       # browser flow, easiest

    or, with an SSH key:

      ssh-keygen -t ed25519 -C "uifabiannn@gmail.com"
      cat ~/.ssh/id_ed25519.pub              # paste at github.com/settings/keys
      git remote set-url origin git@github.com:Fabian-n8n/AI-Quant.git

HELP
  exit 1
fi

# --- 4. Vercel ---------------------------------------------------------------
say "Deploying to Vercel"
if ! command -v vercel >/dev/null 2>&1; then
  warn "Vercel CLI missing. Install it: npm i -g vercel"
  exit 1
fi

if ! vercel whoami >/dev/null 2>&1; then
  warn "Vercel is not authenticated. Run 'vercel login', then re-run this script."
  exit 1
fi

vercel --prod --yes
say "Done. The URL Vercel printed above is your dashboard."
