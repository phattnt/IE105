"""Run the remaining exp3 pilots on a Kaggle GPU notebook (no VM recycling, sessions up to ~12 h):
two other backbones with strategy head, and CodeBERT with strategies slide and slice.

Notebook settings: Accelerator = GPU T4 (x2 if offered), Internet = On. One cell is enough:

    !git clone -q https://github.com/phattnt/IE105.git
    !python IE105/experiments/exp3_pilot_train/kaggle_backbones.py

Same subset, recipe and evaluation as train_pilot.py on Colab. One worker per GPU takes runs from the
queue, so with T4 x2 two runs go in parallel. Results end up in the notebook Output under exp3/results/<run>/.
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = f"{REPO}/experiments"
WORK = "/kaggle/working"
TMP = "/kaggle/tmp" if os.path.isdir("/kaggle") else "/tmp/kaggle_tmp"
# (strategy, model, results name); longest first so the two GPUs finish at about the same time
RUNS = [("slide", "microsoft/codebert-base", "slide"),
        ("head", "neulab/codebert-c", "head_codebert-c"),
        ("head", "microsoft/unixcoder-base", "head_unixcoder-base"),
        ("slice", "microsoft/codebert-base", "slice")]
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

todo = [r for r in RUNS if not os.path.exists(f"/content/exp3/results/{r[2]}/results.json")]
print("runs:", [r[2] for r in todo], "| GPUs:", n_gpu, flush=True)
queue = list(todo)
running = {}            # gpu -> (name, Popen)
T0 = time.time()
last_report = 0
while queue or running:
    for gpu in range(n_gpu):
        if gpu in running and running[gpu][1].poll() is not None:
            name, p = running.pop(gpu)
            print(f"[{(time.time() - T0) / 60:6.1f} min] {name} finished, exit {p.returncode}", flush=True)
            sh(f"tail -3 /content/exp3/logs/{name}.log", check=False)
        if gpu not in running and queue:
            strategy, model, name = queue.pop(0)
            cmd = (f"cd /content/exp1/code && CUDA_VISIBLE_DEVICES={gpu} {sys.executable} train_pilot.py "
                   f"--strategy {strategy} --model {model} --name {name} > /content/exp3/logs/{name}.log 2>&1")
            print(f"[{(time.time() - T0) / 60:6.1f} min] gpu {gpu} <- {name}", flush=True)
            running[gpu] = (name, subprocess.Popen(cmd, shell=True))
    if time.time() - last_report > 300:
        last_report = time.time()
        for gpu, (name, _) in running.items():
            tail = subprocess.run(f"tail -1 /content/exp3/logs/{name}.log", shell=True, capture_output=True, text=True).stdout.strip()
            print(f"[{(time.time() - T0) / 60:6.1f} min] gpu {gpu} {name}: {tail[:160]}", flush=True)
    time.sleep(20)
# keep the Output small: results + logs + best weights only
sh("rm -f /content/exp3/ckpt/*_last.pt", check=False)
sh("du -sh /content/exp3/*", check=False)
