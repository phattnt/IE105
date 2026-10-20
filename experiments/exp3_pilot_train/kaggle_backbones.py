"""Run the exp3 backbone pilots on a Kaggle GPU notebook (no VM recycling, sessions up to ~12 h).

Notebook settings: Accelerator = GPU T4 (x2 if offered), Internet = On. One cell is enough:

    !git clone -q https://github.com/phattnt/IE105.git
    !python IE105/experiments/exp3_pilot_train/kaggle_backbones.py

Same subset, recipe and evaluation as train_pilot.py on Colab (strategy head). With two GPUs the two
backbones run in parallel, one per GPU. Results end up in the notebook Output under exp3/results/<run>/.
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = f"{REPO}/experiments"
WORK = "/kaggle/working"
TMP = "/kaggle/tmp" if os.path.isdir("/kaggle") else "/tmp/kaggle_tmp"
RUNS = [("neulab/codebert-c", "head_codebert-c"), ("microsoft/unixcoder-base", "head_unixcoder-base")]
PRIMEVUL_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        sys.exit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


# the exp1/exp2 modules expect /content/exp1/{primevul,code} and train_pilot.py writes to /content/exp3:
# data goes to scratch space, results/checkpoints/logs to the saved Output directory
os.makedirs(f"{TMP}/primevul", exist_ok=True)
os.makedirs(f"{WORK}/exp3", exist_ok=True)
os.makedirs("/content/exp1/code", exist_ok=True)
for link, target in (("/content/exp1/primevul", f"{TMP}/primevul"), ("/content/exp3", f"{WORK}/exp3")):
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(target, link)
for f in ("exp1_linevul_on_primevul/eval_linevul_primevul.py", "exp2_context_strategies/eval_context_strategies.py",
          "exp3_pilot_train/train_pilot.py"):
    sh(f"cp {EXP}/{f} /content/exp1/code/")
os.makedirs("/content/exp3/logs", exist_ok=True)

sh(f"{sys.executable} -m pip install -q gdown")
if not any(f.endswith(".jsonl") for f in os.listdir(f"{TMP}/primevul")):
    sh(f"gdown -q --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {TMP}/primevul --remaining-ok")
sh(f"ls -la {TMP}/primevul")

# recent PyTorch wheels dropped Pascal (P100, sm_60); fall back to a build that still has it
import torch  # noqa: E402
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
try:
    (torch.ones(2, device="cuda") * 2).sum().item()
except Exception as exc:  # "no kernel image is available for execution on the device"
    print("GPU not supported by this torch build:", exc, "-> installing torch 2.5.1+cu121", flush=True)
    sh(f"{sys.executable} -m pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
n_gpu = max(1, torch.cuda.device_count())

todo = [(m, n) for m, n in RUNS if not os.path.exists(f"/content/exp3/results/{n}/results.json")]
procs = []
T0 = time.time()
for i, (model, name) in enumerate(todo):
    gpu = i % n_gpu
    cmd = (f"cd /content/exp1/code && CUDA_VISIBLE_DEVICES={gpu} {sys.executable} train_pilot.py --strategy head --model {model} "
           f"> /content/exp3/logs/{name}.log 2>&1")
    print(f"[gpu {gpu}] {name}", flush=True)
    procs.append((name, subprocess.Popen(cmd, shell=True)))
    if n_gpu == 1:           # one GPU: run sequentially
        procs[-1][1].wait()
while any(p.poll() is None for _, p in procs):
    time.sleep(300)
    for name, _ in procs:
        tail = subprocess.run(f"tail -1 /content/exp3/logs/{name}.log", shell=True, capture_output=True, text=True).stdout.strip()
        print(f"[{(time.time() - T0) / 60:6.1f} min] {name}: {tail[:160]}", flush=True)
for name, p in procs:
    print(name, "exit", p.returncode, flush=True)
    sh(f"tail -3 /content/exp3/logs/{name}.log", check=False)
# keep the Output small: results + logs + best weights only
sh("rm -f /content/exp3/ckpt/*_last.pt", check=False)
sh("du -sh /content/exp3/*", check=False)
