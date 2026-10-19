@echo off
cd /d %~dp0
title 自认知外挂测试服务

rem ===== 排他性监测（两层）=====
rem 第一层：端口监听。8600 已被占用 → 服务在运行（含手动 python 启动的情况）。
netstat -ano | findstr /R "8600.*LISTENING" >nul
if %errorlevel%==0 goto ALREADY

rem 第二层：原子锁。mkdir 是原子操作——从脚本启动瞬间就占位，
rem 堵住"双击太快、两个窗口都通过端口检查"的竞态窗口。
rem 锁存在 = 另一个实例正在启动中，或上次异常退出未清理。
if exist ".server.lock" (
    echo.
    echo [提示] 另一个实例正在启动中，或服务上次未正常关闭。
    echo        若确认没有服务在运行，请删除 .server.lock 文件夹后重试。
    pause
    exit /b
)
mkdir ".server.lock" 2>nul
if %errorlevel% neq 0 goto ALREADY

echo 正在启动服务，浏览器将自动打开 http://localhost:8600 ...
start "" http://localhost:8600
python server.py --port 8600
rd /s /q ".server.lock" 2>nul
pause
exit /b

:ALREADY
echo.
echo [提示] 8600 端口已有服务在运行，本次不重复启动。
echo        如需重启，请先关闭旧的服务窗口，或结束占用 8600 端口的进程。
start "" http://localhost:8600
pause
exit /b
