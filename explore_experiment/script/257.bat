@echo off
cd /d "%~dp0"

echo --- エンジン起動テスト ---
rem Pythonの場所を確認
where python
if %errorlevel% neq 0 (
    echo [エラー] Python自体が見つかりません。
    pause
    exit
)

echo --- プログラム実行開始 ---
python main.py
if %errorlevel% neq 0 (
    echo [エラー] Python実行中にエラーが発生しました。
    echo 上記のエラーメッセージを確認してください。
    pause
)