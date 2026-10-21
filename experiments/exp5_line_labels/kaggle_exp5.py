"""Kaggle runner for experiment 5. Attach the output of the `ie105-secvul-lines-data` notebook as input
(kernel_sources) and set MODE below:
  smoke  CPU: line-level controls + a tiny ModernBERT run (40 functions, 1 epoch) and PrimeVul scoring on 10 pairs
  train  GPU: each run in RUNS is trained on SecVulEval, then scored on PrimeVul; one run per GPU
"""
import os
import subprocess
import sys
import time

MODE = "smoke"
# (strategy, model, max_len, train_tokens)
RUNS = [("slide", "answerdotai/ModernBERT-base", 2048, 8192)]

subprocess.run("git clone -q https://github.com/phattnt/IE105.git /kaggle/working/IE105 || (cd /kaggle/working/IE105 && git pull -q)", shell=True)
D = "/kaggle/working/IE105/experiments/exp5_line_labels"
subprocess.run(f"{sys.executable} -m pip install -q gdown 'transformers>=4.48'", shell=True)
os.makedirs("/kaggle/working/exp5/logs", exist_ok=True)


def name(strat, model, L):
    return f"{strat}_{model.split('/')[-1]}_{L}_s0"


def job(strat, model, L, toks, extra="", gpu=None, log=None):
    out = f"/kaggle/working/exp5/{name(strat, model, L)}{extra and '_smoke'}"
    cmd = (f"{sys.executable} {D}/train_lines.py --strategy {strat} --model {model} --max_len {L} --train_tokens {toks} --out {out} {extra}"
           f" && {sys.executable} {D}/eval_primevul.py --run {out} {'--limit 10' if extra else ''}")
    env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}))
    print("$", cmd, flush=True)
    return subprocess.Popen(cmd + (f" > {log} 2>&1" if log else ""), shell=True, env=env)


if MODE == "smoke":
    subprocess.run(f"{sys.executable} {D}/train_lines.py --strategy controls", shell=True)
    for strat, model, L, toks in RUNS:
        job(strat, model, L, 4096, extra="--limit 40 --epochs 1 --eval_tokens 8192").wait()
else:
    import torch
    n = max(1, torch.cuda.device_count())
    queue, running, T0 = list(RUNS), {}, time.time()
    while queue or running:
        for g in range(n):
            if g in running and running[g][1].poll() is not None:
                nm, p = running.pop(g)
                print(f"[{(time.time() - T0) / 60:.0f} min] {nm} exit {p.returncode}", flush=True)
                subprocess.run(f"tail -5 /kaggle/working/exp5/logs/{nm}.log", shell=True)
            if g not in running and queue:
                strat, model, L, toks = queue.pop(0)
                nm = name(strat, model, L)
                running[g] = (nm, job(strat, model, L, toks, gpu=g, log=f"/kaggle/working/exp5/logs/{nm}.log"))
        time.sleep(30)
subprocess.run("rm -rf /kaggle/working/IE105 /kaggle/working/exp5/*/ckpt_last.pt", shell=True)
subprocess.run("du -sh /kaggle/working/exp5/*", shell=True)
