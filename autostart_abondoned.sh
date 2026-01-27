#!/bin/bash
sleep 10
# 获取脚本所在目录的绝对路径，实现路径无关性，适配不同主机（setsuna/phoenix）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    -d \
    -m \
    bash -c "./watchdog.sh"
