import subprocess, os, sys
def sh(c):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-2000:], r.stderr[-2000:], flush=True); return r.returncode
sh("rm -rf /content/IE105 && git clone -q https://github.com/phattnt/IE105.git /content/IE105 && cd /content/IE105 && git log --oneline -1")
sh(f"{sys.executable} -m pip install -q gdown datasets")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__, flush=True)
os.makedirs("/content/exp5/logs", exist_ok=True)
sh(f"cd /content/exp5 && {sys.executable} /content/IE105/experiments/exp5_line_labels/prepare_secvuleval.py --out /content/exp5/secvul_lines --primevul_dir /content/exp5/primevul > /content/exp5/logs/prepare.log 2>&1; tail -30 /content/exp5/logs/prepare.log")
