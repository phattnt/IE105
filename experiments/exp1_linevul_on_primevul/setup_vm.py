"""Run ON the Colab VM: download PrimeVul (original release) and the LineVul checkpoint,
then print a short description of the files so the evaluation script can be written
against the real format.
"""
import json
import os
import subprocess
import sys

ROOT = "/content/exp1"
DATA = f"{ROOT}/primevul"
CKPT = f"{ROOT}/ckpt"
os.makedirs(DATA, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

PRIMEVUL_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"   # original release (README of DLVulDet/PrimeVul)
LINEVUL_CKPT = "1oodyQqRb9jEcvLMVVKILmu8qHyNwd-zH"       # 12heads_linevul_model.bin (LineVul README)


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode


sh(f"{sys.executable} -m pip install -q gdown")

if not any(f.endswith(".jsonl") for f in os.listdir(DATA)):
    sh(f"cd {DATA} && gdown --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {DATA} --remaining-ok")
ckpt_path = f"{CKPT}/12heads_linevul_model.bin"
if not os.path.exists(ckpt_path):
    sh(f"gdown https://drive.google.com/uc?id={LINEVUL_CKPT} -O {ckpt_path}")

print("\n=== files ===")
for d in (DATA, CKPT):
    for dp, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(dp, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}")

print("\n=== jsonl peek ===")
for dp, _, fs in os.walk(DATA):
    for f in sorted(fs):
        if not f.endswith(".jsonl"):
            continue
        p = os.path.join(dp, f)
        n = 0
        first = None
        targets = {}
        with open(p) as fh:
            for line in fh:
                n += 1
                js = json.loads(line)
                if first is None:
                    first = js
                t = js.get("target")
                targets[t] = targets.get(t, 0) + 1
        print(f"\n{f}: {n} rows, targets={targets}")
        print("keys:", list(first.keys()))
        for k, v in first.items():
            s = v if isinstance(v, (int, float)) else str(v)[:120].replace("\n", "\\n")
            print(f"  {k}: {s}")
