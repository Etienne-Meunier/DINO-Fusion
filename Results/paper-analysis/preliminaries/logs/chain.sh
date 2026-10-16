#!/bin/zsh
# Wait for the process $1 to exit, then run the grids "grid[:seeds]" one after the other
cd /Users/emeunier/Desktop/Projets/DINO-Fusion/Results/paper-analysis/preliminaries
pid=$1; shift
while kill -0 $pid 2>/dev/null; do sleep 30; done
for spec in "$@"; do
  g=${spec%%:*}; seeds=0; [[ $spec == *:* ]] && seeds=${spec#*:}
  python sweep.py --grid $g --seeds ${(s:,:)seeds} > logs/$g.log 2>&1
done
