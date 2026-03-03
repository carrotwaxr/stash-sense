#!/usr/bin/env bash
# Validate that all version files contain the same version string.
# Usage: ./scripts/check-version.sh [expected-version]
# If expected-version is given (e.g. from a git tag), also checks that files match it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Extract versions from each file
main_ver=$(grep -oP 'version="\K[^"]+' "$ROOT/api/main.py")
settings_ver=$(grep -oP '_version: str = "\K[^"]+' "$ROOT/api/settings_router.py")
plugin_ver=$(grep -oP '^version: \K.+' "$ROOT/plugin/stash-sense.yml")

echo "api/main.py:            $main_ver"
echo "api/settings_router.py: $settings_ver"
echo "plugin/stash-sense.yml: $plugin_ver"

errors=0

if [[ "$main_ver" != "$settings_ver" ]]; then
  echo "ERROR: main.py ($main_ver) != settings_router.py ($settings_ver)"
  errors=1
fi

if [[ "$main_ver" != "$plugin_ver" ]]; then
  echo "ERROR: main.py ($main_ver) != stash-sense.yml ($plugin_ver)"
  errors=1
fi

# If an expected version was passed (e.g. from tag), check against it
if [[ "${1:-}" != "" ]]; then
  expected="$1"
  echo "Expected version:       $expected"
  if [[ "$main_ver" != "$expected" ]]; then
    echo "ERROR: Files have $main_ver but expected $expected"
    errors=1
  fi
fi

if [[ $errors -eq 0 ]]; then
  echo "OK: All version files match ($main_ver)"
else
  exit 1
fi
