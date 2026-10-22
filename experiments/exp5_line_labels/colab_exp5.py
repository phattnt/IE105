"""Colab runner for experiment 5 (runs ON the Colab VM, GPU L4, Colab Pro).

Sets up the VM (repo, packages, PrimeVul, SecVulEval line labels) and then works through the queue RUNS one run
at a time: train_lines.py (SecVulEval) -> eval_primevul.py (PrimeVul). Everything lives under /content/exp5:
  /content/exp5/<run>/            results.json, test_scores.jsonl, primevul_results.json, ckpt_best.pt, ...
  /content/exp5/logs/<run>.log    full log of one run (train + PrimeVul scoring)
  /content/exp5/logs/queue.log    one line per step, every checkpoint save, exit codes; ALL_DONE at the end
A run whose results.json exists is not trained again, and one whose primevul_results.json exists is skipped
entirely, so the script can simply be relaunched after a VM reset or a crash. If /content/exp5/<run>/ckpt_last.pt
exists (left over, or uploaded back from a copy pulled earlier), train_lines.py resumes from it.

The CLI connection drops after ~20 minutes, so launch the script detached from the kernel (see README.md):

    colab new -s exp5 --gpu L4
    colab upload -s exp5 experiments/exp5_line_labels/colab_exp5.py /content/colab_exp5.py
    echo 'import subprocess, sys; subprocess.Popen([sys.executable, "/content/colab_exp5.py"],
          stdout=open("/content/exp5_launch.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)' | colab exec -s exp5

Options: --only NAME [NAME ...] (subset of the queue), --dry (print the commands only), --skip_setup.
The code that is trained is the GitHub main branch (cloned/pulled here): push local changes first.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = "/content/exp5"
REPO = "/content/IE105"
REPO_URL = "https://github.com/phattnt/IE105.git"
D = f"{REPO}/experiments/exp5_line_labels"
DATA = f"{ROOT}/secvul_lines"
PRIMEVUL = f"{ROOT}/primevul"
LOGS = f"{ROOT}/logs"
PRIMEVUL_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"

MODERNBERT, UNIXCODER = "answerdotai/ModernBERT-base", "microsoft/unixcoder-base"
COMMON = "--strategy slide --train_tokens 8192 --bf16 --epochs 4 --patience 2 --ckpt_every 400"
PLAIN = "--lam_line 0 --lam_fn 0"
CONTRAST_LIGHT = "--lam_line 0.3 --lam_fn 1.0 --margin 1"

# (seed, tag, loss args, model, max_len). Run name = slide_{model}_{max_len}_s{seed}{tag}, as in results/.
# Done: ModernBERT 2048 plain s0 (_conv), contrast s0 (_contrast_conv), light contrast s0 (_contrast03_conv).
# 22/10: backbone/window check — UniXcoder (RoBERTa, 1024 positions, code-pretrained with AST) on the same
# line-label pipeline. Its group-A numbers (functions that fit any window) isolate the backbone effect from the
# window effect when set against ModernBERT 2048.
RUNS = [
    (0, "_conv", PLAIN, UNIXCODER, 1024),
]
EXTRA_SEEDS = [
    (1, "_conv", PLAIN, MODERNBERT, 2048),
    (1, "_contrast03_conv", CONTRAST_LIGHT, MODERNBERT, 2048),
    (2, "_conv", PLAIN, MODERNBERT, 2048),
    (2, "_contrast03_conv", CONTRAST_LIGHT, MODERNBERT, 2048),
]
if os.environ.get("EXP5_EXTRA_SEEDS"):
    RUNS += EXTRA_SEEDS

T0 = time.time()


def qlog(*a):
    msg = " ".join(str(x) for x in a)
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(f"{LOGS}/queue.log", "a") as fh:
        fh.write(line + "\n")


def sh(cmd, check=True):
    qlog("$", cmd)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        qlog(f"FAILED ({r.returncode}): {cmd}")
        sys.exit(r.returncode)
    return r.returncode


def run_name(seed, tag, model, max_len):
    return f"slide_{model.split('/')[-1]}_{max_len}_s{seed}{tag}"


def setup():
    if os.path.isdir(f"{REPO}/.git"):
        sh(f"git -C {REPO} pull -q")
    else:
        sh(f"git clone -q {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log -1 --oneline", check=False)
    sh(f"{sys.executable} -m pip install -q 'transformers>=4.48' gdown datasets scikit-learn")
    os.makedirs(PRIMEVUL, exist_ok=True)
    need = [f"primevul_{s}.jsonl" for s in ("test", "test_paired")]
    if not all(os.path.exists(f"{PRIMEVUL}/{f}") for f in need):
        # gdown >= 5 dropped --remaining-ok; the folder holds only a few files anyway
        sh(f"gdown -q --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {PRIMEVUL}", check=False)
        for p in glob.glob(f"{PRIMEVUL}/**/primevul_*.jsonl", recursive=True):   # flatten a nested folder
            if os.path.dirname(p) != PRIMEVUL:
                shutil.move(p, f"{PRIMEVUL}/{os.path.basename(p)}")
        missing = [f for f in need if not os.path.exists(f"{PRIMEVUL}/{f}")]
        if missing:
            qlog("PrimeVul download incomplete, missing", missing)
            sys.exit(1)
    if not os.path.exists(f"{DATA}/test.jsonl"):
        sh(f"{sys.executable} {D}/prepare_secvuleval.py --out {DATA} --primevul_dir {PRIMEVUL}")


def stream(cmd, log_path):
    """Run cmd, append its output to log_path, and copy checkpoint / epoch / error lines to the queue log."""
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {cmd}\n")
        fh.flush()
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            if any(k in line for k in ("saved checkpoint", "saved best", "RESUMED", "valid ep", "early stop", "Error", "DONE")):
                qlog("   ", line.strip()[:200])
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="run names (or substrings) to keep from the queue")
    ap.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--skip_setup", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    queue = [(run_name(s, t, m, L), s, t, f"--model {m} --max_len {L} {x}") for s, t, x, m, L in RUNS]
    if a.only:
        queue = [q for q in queue if any(o in q[0] for o in a.only)]
    qlog("queue:", ", ".join(q[0] for q in queue))
    if a.dry:
        for nm, s, t, x in queue:
            print(f"{sys.executable} {D}/train_lines.py {COMMON} {x} --seed {s} --tag {t} --data {DATA} --out {ROOT}/{nm}")
            print(f"{sys.executable} {D}/eval_primevul.py --run {ROOT}/{nm} --primevul {PRIMEVUL}")
        return
    import torch  # preinstalled on Colab
    if not torch.cuda.is_available():
        qlog("no GPU visible - start the session with `colab new --gpu L4`; stopping")
        sys.exit(1)
    qlog("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
    if not a.skip_setup:
        setup()
    for nm, s, t, x in queue:
        out, log = f"{ROOT}/{nm}", f"{LOGS}/{nm}.log"
        if os.path.exists(f"{out}/primevul_results.json"):
            qlog(f"skip {nm} (trained and scored)")
            continue
        t_run = time.time()
        if os.path.exists(f"{out}/results.json"):
            qlog(f"{nm}: results.json exists, PrimeVul scoring only")
        else:
            qlog(f"start {nm}")
            rc = stream(f"{sys.executable} {D}/train_lines.py {COMMON} {x} --seed {s} --tag {t} --data {DATA} --out {out}", log)
            if rc != 0 or not os.path.exists(f"{out}/results.json"):
                qlog(f"{nm}: train exit {rc} - moving on (relaunch resumes from ckpt_last.pt)")
                continue
        rc = stream(f"{sys.executable} {D}/eval_primevul.py --run {out} --primevul {PRIMEVUL}", log)
        qlog(f"{nm}: done (PrimeVul exit {rc}) in {(time.time() - t_run) / 3600:.2f} h")
    qlog("ALL_DONE")
    open(f"{LOGS}/ALL_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
