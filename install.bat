@echo off
REM Build the release binary from this folder.
cd /d "%~dp0"
cargo build --release

IF ERRORLEVEL 1 (
  echo.
  echo [ERROR] Build failed.
) ELSE (
  echo.
  echo [OK] Built: "%CD%\target\release\ollama-rename.exe"
)

echo.
echo Press any key to close...
pause >nul
