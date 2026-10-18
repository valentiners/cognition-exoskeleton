@echo off
chcp 65001 >nul
cd /d %~dp0
title 自认知外挂测试服务

:: 排他性判断：8600 端口已被占用说明服务已在运行，禁止重复启动
netstat -ano | findstr /R "8600.*LISTENING" >nul
if %errorlevel%==0 (
  echo.
  echo [提示] 检测到 8600 端口已有服务在运行，本次不重复启动。
  echo        如需重启，请先关闭旧的服务窗口，或结束占用进程后再运行本脚本。
  echo        正在为你打开页面 ...
  start "" http://localhost:8600
  pause
  exit /b
)

echo 正在启动服务，浏览器将自动打开 http://localhost:8600 ...
start "" http://localhost:8600
python server.py --port 8600
pause
