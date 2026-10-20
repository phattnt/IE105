#!/bin/bash
# Runs ON the Colab VM, detached from the CLI connection: python code/run_vm.sh-equivalent.
# Each strategy is skipped if its results.json already exists, so the script can be relaunched after a VM reset.
mkdir -p /content/exp3/logs /content/exp3/results
cd /content/exp1/code
for s in length head slide slice; do
  if [ -f /content/exp3/results/$s/results.json ]; then echo "skip $s"; continue; fi
  python train_pilot.py --strategy $s > /content/exp3/logs/$s.log 2>&1
done
echo ALL_DONE > /content/exp3/logs/ALL_DONE
