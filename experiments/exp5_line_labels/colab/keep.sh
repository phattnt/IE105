#!/bin/bash
# Keep a client connection open to the kernel at all times (Colab Pro reclaims ~20 min after the last client disconnects).
# Pauses while relaunch.sh runs (its own execs keep the connection, and keep cells would block its setup cells).
export PATH="$HOME/.local/bin:$PATH"; S=${S:-$(cd "$(dirname "$0")" && pwd)/work_conv2}
while ! grep -q "ALL FINISHED" $S/events.log 2>/dev/null; do
  if pgrep -f "conv2/[r]elaunch.sh" >/dev/null; then sleep 20; continue; fi
  timeout 900 colab exec -s conv2 -f $S/keep.py >> $S/keep.log 2>&1
  echo "== $(date -u +%H:%M) keep exec ended rc=$?" >> $S/keep.log; sleep 20
done
