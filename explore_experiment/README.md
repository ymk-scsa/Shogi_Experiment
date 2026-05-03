'''
C:.
│  .python-version
│  main.py
│
├─model
│      activation_function.py
│      block.py
│      model.py
│      __init__.py
│
├─player
│      base_player.py
│      mcts_player.py
│      npls_node.py
│      npls_player.py
│      uct_node.py
│      __init__.py
│
├─script
│      MCTS_ResNet.bat
│      NPLS_ResNet.bat
│
├─shogi
│      feature.py
│      __init__.py
│
├─train
│      train.py
│      __init__.py
│
└─util
        dataloader.py
        datawriter.py
        directory.py
        logger.py
        __init__.py
'''

scp -r checkpoint E2141701@gpumng2.cle.it-chiba.ac.jp:~/explore_experiment/

@echo off
rem 1. バッチファイルがある場所の1つ上のフォルダ（プロジェクトルート）に移動
cd /d "%~dp0.."

rem 2. Pythonの実行。システムのpyランチャーを使用して、Microsoft Storeを回避します
py remote/usi_proxy.py --host 100.86.252.25 --port 49001 --token Nigohachi257