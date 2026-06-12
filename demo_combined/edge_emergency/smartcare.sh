#!/usr/bin/env bash
# Start / stop the Smart-Care edge node on the Pi.
#   smartcare.sh start | stop | restart | status
# Runs detached, so it keeps running after you disconnect SSH.
cd "$(dirname "$0")/../.." || exit 1          # repo root
PY=.venv-edge/bin/python
NODE=demo_combined/edge_emergency/edge_node.py
# Edit these flags to taste. --max-fps 0 = uncapped; set e.g. 15 if the Pi runs hot.
ARGS="--camera 4 --pose-backend movenet_multi --num-person 2 --short-side 320 \
      --recog-every 2 --max-fps 0 --port 8443 --https --http-port 8000"
PATTERN='[e]dge_node.py'                        # bracket = pkill won't match itself

case "${1:-}" in
  start)
    if pgrep -f "$PATTERN" >/dev/null; then echo "already running"; exit 0; fi
    PYTHONIOENCODING=utf-8 nohup setsid "$PY" -u $NODE $ARGS \
      > /tmp/edge.log 2>&1 < /dev/null & disown
    sleep 1
    echo "started -> https://$(hostname -I | awk '{print $1}'):8443/"
    ;;
  stop)
    pkill -f "$PATTERN" && echo "stopped" || echo "not running"
    ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  status)
    pgrep -f "$PATTERN" >/dev/null && echo "RUNNING" || echo "stopped"
    echo "temp: $(vcgencmd measure_temp 2>/dev/null)"
    ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 1 ;;
esac
