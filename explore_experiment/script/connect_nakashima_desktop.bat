@echo off

@wsl -d Ubuntu-24.04 bash -c "cd /mnt/c/Users/kotet/Documents/program/Shogi_Experiment/explore_experiment && source /mnt/c/Users/kotet/Documents/program/Shogi_Experiment/.venv/bin/activate && python remote/usi_proxy.py --host 100.98.165.126 --port 49001 --token Nigohachi257"
