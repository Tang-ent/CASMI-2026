#!/bin/bash
# Host memory watchdog for local docker runs (added 2026-10-17 after the PC froze with two 20 GB containers on a
# 31 GB host). Every 5 s: log MemAvailable, swap used and each casmi26 container's memory; if MemAvailable drops below
# MIN_MB, stop the most recently started casmi26 container (docker stop; outputs written so far stay, k41 resumes).
#   setsid nohup tools/local/memwatch.sh > /dev/null 2>&1 &     # log in out/memwatch.log
MIN_MB=${MIN_MB:-3000}
LOG=${LOG:-$(cd "$(dirname "$0")/../.." && pwd)/out/memwatch.log}
echo "$(date '+%F %T') start MIN_MB=$MIN_MB" >> "$LOG"
n=0
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  swap=$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {print int((t-f)/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_MB" ]; then
    c=$(docker ps --filter name=casmi26 --format '{{.CreatedAt}}|{{.Names}}' | sort -r | head -1 | cut -d'|' -f2)
    echo "$(date '+%F %T') LOW MemAvailable ${avail} MB swap ${swap} MB -> docker stop ${c:-<none>}" >> "$LOG"
    [ -n "$c" ] && docker stop -t 5 "$c" >> "$LOG" 2>&1
  fi
  n=$((n + 1))
  if [ $((n % 12)) -eq 0 ]; then  # once a minute
    st=$(docker stats --no-stream --format '{{.Name}}={{.MemUsage}}' 2>/dev/null | grep casmi26 | tr '\n' ' ')
    echo "$(date '+%F %T') avail ${avail} MB swap ${swap} MB ${st}" >> "$LOG"
  fi
  sleep 5
done
