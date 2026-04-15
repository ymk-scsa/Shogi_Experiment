'''
@echo off
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%\..

echo --- Shogi Engine Search Selection ---
echo [1] MCTS (Monte Carlo Tree Search)
echo [2] NPLS (Neural Priority List Search)
set /p CHOICE="Select Search Type (1 or 2, default=1): "

set SEARCH_TYPE=mcts
if "%CHOICE%"=="2" set SEARCH_TYPE=npls

echo starting with %SEARCH_TYPE%...
echo.

echo --- Start GNN Shogi Engine ---
:: 仮想環境のPythonで実行
".\venv\Scripts\python.exe" main.py --search_type %SEARCH_TYPE%

echo.
echo --- Engine Stopped ---
pause
'''

'''
NPLS(Neural Priority Leaf Search)
MCTS(Monte Carlo Tree Search)
'''
