# Runs IN the kernel cell. Resumable: train_lines.py resumes from ckpt_last.pt in OUT; eval skipped if already done.
import subprocess, sys, os
D = "/content/IE105/experiments/exp5_line_labels"
OUT = "/content/exp5/conv2"
os.makedirs("/content/exp5/logs", exist_ok=True)
if not os.path.exists(f"{OUT}/results.json"):
    subprocess.run(f"{sys.executable} {D}/train_lines.py --strategy slide --model answerdotai/ModernBERT-base --max_len 2048 "
                   f"--train_tokens 8192 --eval_tokens 32768 --lr 3e-5 --epochs 4 --patience 2 --ckpt_every 400 --bf16 --lam_line 1 --lam_fn 1 --margin 1 --max_pair_views 2 "
                   f"--data /content/exp5/secvul_lines --out {OUT} >> /content/exp5/logs/conv2.log 2>&1; echo TRAIN_EXIT=$? >> /content/exp5/logs/conv2.log", shell=True)
if os.path.exists(f"{OUT}/results.json") and not os.path.exists(f"{OUT}/primevul_results.json"):
    subprocess.run(f"{sys.executable} {D}/eval_primevul.py --run {OUT} --primevul /content/exp5/primevul >> /content/exp5/logs/conv2_primevul.log 2>&1; "
                   f"echo EVAL_EXIT=$? >> /content/exp5/logs/conv2_primevul.log", shell=True)
print("ALL DONE", flush=True)
