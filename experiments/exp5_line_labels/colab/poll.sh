#!/bin/bash
# Mirror logs, pull checkpoints/results as they appear, relaunch on VM loss, stop the VM when everything is done.
export PATH="$HOME/.local/bin:$PATH"
S=${S:-$(cd "$(dirname "$0")" && pwd)/work_conv2}
C=$S/ckpt; EV=$S/events.log
R=${R:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/experiments/exp5_line_labels/results/slide_ModernBERT-base_2048_s0_contrast_conv}
declare -A seen; ck_seen=0; best_seen=0; miss=0; relaunches=0
dl() { timeout ${3:-900} colab download -s conv2 "$1" "$2.part" >/dev/null 2>&1 && mv "$2.part" "$2"; }
while true; do
  got=0
  for f in conv2 conv2_primevul; do
    if dl /content/exp5/logs/$f.log $S/$f.vm.log 60; then
      got=1; n=$(wc -l < $S/$f.vm.log); p=${seen[$f]:-0}; [ "$n" -lt "$p" ] && p=0
      [ "$n" -gt "$p" ] && tail -n +$((p+1)) $S/$f.vm.log | tr '\r' '\n' | grep -v -E "it/s\]|Token indices" | sed "s/^/[$f] /" | tee -a $S/$f.all.log >> $EV; seen[$f]=$n
    fi
  done
  if [ $got = 1 ]; then
    miss=0
    nck=$(grep -c "saved checkpoint" $S/conv2.vm.log 2>/dev/null); nb=$(grep -c "saved best" $S/conv2.vm.log 2>/dev/null)
    if [ "$nck" != "$ck_seen" ] && [ "$nck" -gt 0 ]; then dl /content/exp5/conv2/ckpt_last.pt $C/ckpt_last.pt && echo "[poll] pulled ckpt_last ($(grep 'saved checkpoint' $S/conv2.vm.log | tail -1 | grep -o 'epoch.*'))" >> $EV; ck_seen=$nck; fi
    if [ "$nb" != "$best_seen" ] && [ "$nb" -gt 0 ]; then dl /content/exp5/conv2/ckpt_best.pt $C/ckpt_best.pt && echo "[poll] pulled ckpt_best" >> $EV; best_seen=$nb; fi
    if grep -q "TRAIN_EXIT=" $S/conv2.vm.log && [ ! -f $C/results.json ]; then
      dl /content/exp5/conv2/results.json $C/results.json 120; dl /content/exp5/conv2/test_scores.jsonl $C/test_scores.jsonl
      [ -f $C/results.json ] && echo "[poll] pulled SecVulEval results" >> $EV; fi
    if grep -q "EVAL_EXIT=" $S/conv2_primevul.vm.log 2>/dev/null; then
      mkdir -p $R
      dl /content/exp5/conv2/primevul_results.json $R/primevul_results.json 120; dl /content/exp5/conv2/primevul_test_scores.jsonl $R/primevul_test_scores.jsonl
      cp $C/results.json $C/test_scores.jsonl $R/ 2>/dev/null; cp $S/conv2.all.log $R/train.log; cp $S/conv2_primevul.all.log $R/primevul.log
      timeout 120 colab stop -s conv2 >/dev/null 2>&1; echo "[poll] ALL FINISHED, VM stopped" >> $EV; exit 0
    fi
  else
    miss=$((miss+1)); echo "[poll] $(date -u +%H:%M) no log reachable (#$miss)" >> $EV
    if [ $miss -ge 2 ] && ! timeout 60 colab sessions 2>&1 | grep -q "\[conv2\]"; then
      relaunches=$((relaunches+1)); [ $relaunches -gt 8 ] && { echo "[poll] giving up after 8 relaunches" >> $EV; exit 1; }
      echo "[poll] VM gone -> RELAUNCH #$relaunches" >> $EV; unset seen; declare -A seen
      $S/relaunch.sh >> $S/relaunch.log 2>&1 &
      sleep 900; miss=0
    fi
  fi
  sleep 120
done
