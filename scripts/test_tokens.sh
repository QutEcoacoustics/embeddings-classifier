#!/usr/bin/env bash

set -euo pipefail

check_token() {
  local token="$1"
  local name="$2"
  local url="$3"

  [[ -n "$token" ]] || { echo "$name is missing" >&2; exit 1; }

  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token token=\"${token}\"" \
    "${url}/status")

  [[ "$code" == "200" ]] || { echo "$name invalid or API unavailable (HTTP $code)" >&2; exit 1; }
}

check_token "${BAW_AUTH_TOKEN:-}" "BAW_AUTH_TOKEN" "https://api.ecosounds.org"

echo "Token validation passed"
