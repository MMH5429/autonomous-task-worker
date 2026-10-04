#!/bin/bash
# stop any running mockco, then start one with the given flags
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*mockco/erp.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 1
nohup python mockco/erp.py --port 8099 --reset "$@" >/dev/null 2>&1 &
for i in $(seq 1 15); do curl -s --max-time 2 http://127.0.0.1:8099/health >/dev/null 2>&1 && return 0 2>/dev/null || sleep 1; done
