@echo off
chcp 65001 >nul
cd /d %~dp0
title 自认知外挂测试服务
echo 正在启动服务，浏览器将自动打开 http://localhost:8600 ...
start "" http://localhost:8600
python server.py --port 8600
pause
