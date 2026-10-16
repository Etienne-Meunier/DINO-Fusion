#!/bin/zsh
# Run the sweep grids one after the other, after the running sweep
cd /Users/emeunier/Desktop/Projets/DINO-Fusion/Results/paper-analysis/preliminaries
while pgrep -f "sweep.py --grid calib" >/dev/null; do sleep 30; done
for g in "$@"; do python sweep.py --grid $g > logs/$g.log 2>&1; done
