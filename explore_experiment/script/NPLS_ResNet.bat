@echo off

@wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/kotet/Documents/program/Shogi_Experiment/explore_experiment && source /mnt/c/Users/kotet/Documents/program/Shogi_Experiment/GNN_Experiment_20251229/.venv/bin/activate && /home/kotetsu/.pyenv/shims/python player/npls_player.py"
