#!/usr/bin/env bash
# Build the site, serve it over http (zero-md needs real http, not file://),
# and open it in the platform browser. Ctrl-C stops the server.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="${PORT:-8000}"

"$here/build.sh"

cd "$here/_site"
python3 -m http.server "$port" &
server_pid=$!
cd "$here"

cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

sleep 0.5

url="http://localhost:$port/"
if command -v open >/dev/null; then
  open "$url"
elif command -v xdg-open >/dev/null; then
  xdg-open "$url"
elif command -v wslview >/dev/null; then
  wslview "$url"
else
  echo "open a browser to $url"
fi

wait "$server_pid"
