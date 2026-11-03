#!/bin/zsh
# Rest of queue16 at b8 only : waits for the running nl_c7010, then C7 eta 30, 50, 100 with noisy land and raw noise.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
while pgrep -f "name nl_c7010" > /dev/null; do sleep 10; done
for e in 30 50 100; do
  python run.py --variant c7 --eta $e --eps 8.6e-08 --centred --raw_noise --name nl_c7$(printf %03d $e) > logs/nl_c7$e.log 2>&1
done
