@echo off
title Python Q^&A Chatbot
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel% equ 0 (
    py chatbot.py %*
) else (
    python chatbot.py %*
)
if %errorlevel% neq 0 (
    echo.
    echo If Python did not run, check that Python is installed and added to PATH.
    pause
)
