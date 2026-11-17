#!/bin/sh
# Drop from root to PUID:PGID so files the sidecar creates (database, models,
# subtitles) are owned like the rest of the library. unRAID default: 99:100.
#
#   PUID / PGID  user and group ids to run as (PUID=0 keeps root)
#   UMASK        file creation mask (default 022)
set -eu

PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK="${UMASK:-022}"
DATA_DIR="${DATA_DIR:-/data}"
INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-/root/.insightface}"
APP_HOME=/home/stashsense

umask "$UMASK"

# Already non-root (docker run --user): nothing to switch.
if [ "$(id -u)" != "0" ]; then
    exec "$@"
fi

if [ "$PUID" = "0" ]; then
    echo "[entrypoint] PUID=0: running as root"
    exec "$@"
fi

# Give the ids names; some libraries look the current user up.
getent group "$PGID" >/dev/null || groupadd -o -g "$PGID" stashsense
getent passwd "$PUID" >/dev/null || \
    useradd -o -u "$PUID" -g "$PGID" -d "$APP_HOME" -M -s /usr/sbin/nologin stashsense

# Hand the app's own directories to PUID:PGID. Only files with other owners
# are touched, so this is cheap after the first start. The media library is
# never chowned.
for dir in "$DATA_DIR" "$INSIGHTFACE_HOME" "$APP_HOME"; do
    mkdir -p "$dir"
    find "$dir" \( ! -uid "$PUID" -o ! -gid "$PGID" \) -exec chown -h "$PUID:$PGID" {} +
done

echo "[entrypoint] running as uid=$PUID gid=$PGID umask=$UMASK"
export HOME="$APP_HOME"
exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups -- "$@"
