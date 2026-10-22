#!/bin/bash
# (Re)create the L4 session, restore code/data/checkpoints/results, launch run.py in the kernel.
export PATH="$HOME/.local/bin:$PATH"
S=${S:-$(cd "$(dirname "$0")" && pwd)/work_conv2}
C=$S/ckpt
echo "== $(date -u +%H:%M) relaunch"
timeout 300 colab new -s conv2 --gpu L4 2>&1 | tail -1 | grep -q READY || { echo "could not create session"; exit 1; }
timeout 1500 colab exec -s conv2 -f $S/setup.py 2>&1 | grep -E "train_functions|L4" | head -2
echo 'import os; os.makedirs("/content/exp5/conv2", exist_ok=True); print("dir ok")' > $S/mk.py
timeout 300 colab exec -s conv2 -f $S/mk.py 2>&1 | grep "dir ok"
: > $S/assemble.py
for f in ckpt_last.pt ckpt_best.pt; do
  [ -f $C/$f ] || continue
  rm -rf $S/chunks; mkdir -p $S/chunks; split -b 40m -d $C/$f $S/chunks/$f.part; n=0; t=0
  for part in $S/chunks/$f.part*; do t=$((t+1)); timeout 300 colab upload -s conv2 $part /content/exp5/conv2/$(basename $part) 2>&1 | grep -q Uploaded && n=$((n+1)); done
  echo "$f: $n/$t chunks"
  echo "import glob,os; ps=sorted(glob.glob('/content/exp5/conv2/$f.part*')); open('/content/exp5/conv2/$f','wb').write(b''.join(open(p,'rb').read() for p in ps)); [os.remove(p) for p in ps]; print('$f', os.path.getsize('/content/exp5/conv2/$f'))" >> $S/assemble.py
done
for f in results.json test_scores.jsonl; do [ -f $C/$f ] && timeout 300 colab upload -s conv2 $C/$f /content/exp5/conv2/$f 2>&1 | grep -c Uploaded | sed "s/^/$f: /"; done
[ -s $S/assemble.py ] && timeout 300 colab exec -s conv2 -f $S/assemble.py 2>&1 | grep -E "\.pt [0-9]+"
timeout 300 colab upload -s conv2 $S/run_inner.py /content/exp5/run_inner_conv2.py 2>&1 | grep -c Uploaded | sed 's/^/run_inner: /'
echo "== $(date -u +%H:%M) launch"
timeout 7000 colab exec -s conv2 -f $S/run.py > $S/inkernel.log 2>&1
echo "== $(date -u +%H:%M) exec returned rc=$? (kernel may keep running)"
