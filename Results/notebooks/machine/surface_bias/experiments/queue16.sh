#!/bin/zsh
# Noisy land with raw (uncentred) noise : unconstrained and C7 eta series at b8, seed 0 (stopped after nl_c7010, rest in queue16b.sh).
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
python run.py --variant none --eta 0 --eps 0 --centred --raw_noise --name nl_none > logs/nl_none.log 2>&1
for e in 10 30 50 100; do
  python run.py --variant c7 --eta $e --eps 8.6e-08 --centred --raw_noise --name nl_c7$(printf %03d $e) > logs/nl_c7$e.log 2>&1
done


