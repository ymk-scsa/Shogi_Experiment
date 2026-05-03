@echo off
rem 1. バッチファイルがある場所の1つ上のフォルダ（プロジェクトルート）に移動
cd /d "%~dp0.."

rem 2. Pythonの実行。システムのpyランチャーを使用して、Microsoft Storeを回避します
py remote/usi_proxy.py --host 100.86.252.25 --port 49001 --token Nigohachi257


