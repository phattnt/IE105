"""Kaggle TPU runner for experiment 5 (attach `ie105-secvul-lines-data` as input).
MODE bench: environment report + N timed training steps of the ModernBERT line model (sizes the real run).
MODE train: full run (SecVulEval train -> valid/test -> PrimeVul protocol).
"""
import os
import subprocess
import sys

MODE = "bench"
ARGS = "--strategy slide --model answerdotai/ModernBERT-base --max_len 2048 --epochs 3"


def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


sh("git clone -q https://github.com/phattnt/IE105.git /kaggle/working/IE105 || (cd /kaggle/working/IE105 && git pull -q)")
D = "/kaggle/working/IE105/experiments/exp5_line_labels"
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E '^(torch|torch-xla|torch_xla|jax|jaxlib|transformers|libtpu)' ")
sh("ls /dev | grep -i -E 'accel|vfio' | head; env | grep -i -E 'TPU|XLA|PJRT' | head -20")
try:
    import torch_xla  # noqa: F401
except Exception as e:
    print("torch_xla import failed:", repr(e), flush=True)
    sys.exit(1)
sh(f"{sys.executable} -m pip install -q gdown 'transformers>=4.48'")
os.makedirs("/kaggle/working/exp5", exist_ok=True)
if MODE == "bench":
    sh(f"{sys.executable} {D}/train_lines_xla.py {ARGS} --bench 60 --log_every 5")
else:
    sh(f"{sys.executable} {D}/train_lines_xla.py {ARGS}")
sh("rm -rf /kaggle/working/IE105; du -sh /kaggle/working/exp5/*")
