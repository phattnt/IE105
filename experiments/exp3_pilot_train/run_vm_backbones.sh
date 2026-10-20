#!/bin/bash
# Runs ON the Colab VM after run_vm.sh: same subset, strategy head, other encoders.
while [ ! -f /content/exp3/logs/ALL_DONE ]; do sleep 60; done
cd /content/exp1/code
for m in neulab/codebert-c microsoft/unixcoder-base; do
  n=head_${m#*/}
  if [ -f /content/exp3/results/$n/results.json ]; then echo "skip $n"; continue; fi
  python train_pilot.py --strategy head --model $m > /content/exp3/logs/$n.log 2>&1
done
echo done > /content/exp3/logs/BACKBONES_DONE
