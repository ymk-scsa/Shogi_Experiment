@echo off
rem カレントディレクトリをこのファイルの場所に移動
cd /d %~dp0

rem 仮想環境を有効化
call venv\Scripts\activate

rem pythonでメインプログラムを実行
rem 現状は引数なしでOKです
python main.py

pause