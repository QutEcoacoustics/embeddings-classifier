#!/usr/bin/env bash

set -euo pipefail

# Print next CalVer version for this repo, or print nothing when there are no
# releasable changes since the last tag. This repo's canonical version lives in
# src/VERSION.

latest_tag="$(git tag --list 'v*' --sort=-creatordate | head -n 1 || true)"

if [[ -n "${latest_tag}" ]]; then
  if git diff --quiet "${latest_tag}"..HEAD -- . ':(exclude)src/VERSION'; then
    # No releasable changes since last tag.
    exit 0
  fi
fi

today="$(date -u +%Y.%m.%d)"
latest_version="${latest_tag#v}"

if [[ "${latest_version}" =~ ^${today}(\.([0-9]+))?$ ]]; then
  suffix="${BASH_REMATCH[2]:-}"
  if [[ -z "${suffix}" ]]; then
    echo "${today}.1"
  else
    echo "${today}.$((suffix + 1))"
  fi
else
  echo "${today}"
fi
