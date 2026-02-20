#!/bin/bash
# Builds a Stash plugin index for distribution via GitHub Pages.
# Output: <outdir>/index.yml and <outdir>/stash-sense.zip
#
# Usage: ./build_plugin_index.sh [outdir]

set -e

outdir="${1:-_plugin_site}"
rm -rf "$outdir"
mkdir -p "$outdir"

plugin_dir="plugin"
plugin_id="stash-sense"

# Extract metadata from plugin YAML
name=$(grep "^name:" "$plugin_dir/$plugin_id.yml" | head -n 1 | cut -d' ' -f2- | sed -e 's/\r//')
description=$(grep "^description:" "$plugin_dir/$plugin_id.yml" | head -n 1 | cut -d' ' -f2- | sed -e 's/\r//')
yml_version=$(grep "^version:" "$plugin_dir/$plugin_id.yml" | head -n 1 | cut -d' ' -f2- | sed -e 's/\r//')

# Get git info
commit_hash=$(git log -n 1 --pretty=format:%h -- "$plugin_dir"/*)
updated=$(TZ=UTC0 git log -n 1 --date="format-local:%F %T" --pretty=format:%ad -- "$plugin_dir"/*)
version="$yml_version-$commit_hash"

# Create zip
zipfile=$(realpath "$outdir/$plugin_id.zip")
pushd "$plugin_dir" > /dev/null
zip -r "$zipfile" . -x "__pycache__/*" "*.pyc" > /dev/null
popd > /dev/null

sha=$(sha256sum "$zipfile" | cut -d' ' -f1)

# Write index.yml
cat > "$outdir/index.yml" <<EOF
- id: $plugin_id
  name: $name
  metadata:
    description: $description
  version: $version
  date: $updated
  path: $plugin_id.zip
  sha256: $sha
EOF

echo "Built plugin index: $outdir/index.yml (version: $version)"
