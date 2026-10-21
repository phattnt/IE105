"""Kaggle runner for experiment 5. Attach the output of the `ie105-secvul-lines-data` notebook as input
(kernel_sources) and set MODE below (the push script rewrites it):
  smoke  CPU: line-level controls on the full test split + 1-epoch training on 60 functions per split (code check)
  train  GPU: the line-level models listed in RUNS, one per GPU
"""
import os
import subprocess
import sys

MODE = "smoke"
RUNS = [("head", "neulab/codebert-c", 512), ("slide", "neulab/codebert-c", 512)]

subprocess.run("git clone -q https://github.com/phattnt/IE105.git /kaggle/working/IE105 || (cd /kaggle/working/IE105 && git pull -q)", shell=True)
S = "/kaggle/working/IE105/experiments/exp5_line_labels/train_lines.py"


def run(args, gpu=None, log=None):
    env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}))
    cmd = f"{sys.executable} {S} {args}" + (f" > {log} 2>&1" if log else "")
    print("$", cmd, flush=True)
    return subprocess.Popen(cmd, shell=True, env=env)


os.makedirs("/kaggle/working/exp5/logs", exist_ok=True)
if MODE == "smoke":
    run("--strategy controls").wait()
    for strat in ("head", "slide"):
        run(f"--strategy {strat} --limit 60 --epochs 1 --bs 4 --eval_bs 8 --out /kaggle/working/exp5/smoke_{strat}").wait()
else:
    import torch
    n = max(1, torch.cuda.device_count())
    queue, running = list(RUNS), {}
    import time
    while queue or running:
        for g in range(n):
            if g in running and running[g].poll() is not None:
                running.pop(g)
            if g not in running and queue:
                strat, model, L = queue.pop(0)
                running[g] = run(f"--strategy {strat} --model {model} --max_len {L}", gpu=g,
                                 log=f"/kaggle/working/exp5/logs/{strat}_{model.split('/')[-1]}_{L}.log")
        time.sleep(30)
subprocess.run("rm -rf /kaggle/working/IE105 /kaggle/working/exp5/*/ckpt_last.pt", shell=True)
subprocess.run("du -sh /kaggle/working/exp5/*", shell=True)
