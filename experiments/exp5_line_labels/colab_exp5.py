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
SYNTH = f"{ROOT}/synth_guards"
PV_LINES = f"{ROOT}/primevul_lines"
DATA_CLEAN = f"{ROOT}/secvul_lines_clean"   # train/valid without trivial flaw labels and dense (>0.5) rewrites; same test
SYNTH_PAIRS = 40000
COMMON = "--strategy slide --train_tokens 8192 --bf16 --ckpt_every 400"
PLAIN = "--lam_line 0 --lam_fn 0"
CONTRAST = "--lam_line 1.0 --lam_fn 1.0 --margin 1"
CONTRAST_LIGHT = "--lam_line 0.3 --lam_fn 1.0 --margin 1"

# One dict per run. Run name = slide_{model}_{max_len}_s{seed}{tag}, as in results/. Keys: seed, tag, loss, model,
# max_len, and optionally bf16 (default True; False for GPUs without bf16 such as T4), data (default: SecVulEval line labels), epochs/patience (default 4/2), init (name of an
# earlier run in this queue whose ckpt_best.pt initialises the model), extra (more train_lines.py flags).
# Done: ModernBERT 2048 plain s0 (_conv), contrast s0 (_contrast_conv), light contrast s0 (_contrast03_conv),
# UniXcoder 1024 plain s0 (_conv).
# 22/10 Train 13: stage 1 learns "a guard is missing above this line" from synthetic counterfactual pairs
# (make_synthetic_guards.py: one guard removed from a benign PrimeVul function; both versions differ by 1-3 lines, so
# no length shortcut exists); stage 2 fine-tunes that model on SecVulEval like the plain _conv run, once as is and once
# with the pair length distribution balanced. Epoch selection uses Top-5 + length-debiased pair accuracy.
SELECT = "--select top5_pair_debias"
STAGE1 = "slide_ModernBERT-base_2048_s0_synth1"
RUNS = [
    dict(seed=0, tag="_synth1", loss=CONTRAST, model=MODERNBERT, max_len=2048, data=SYNTH, epochs=1, patience=0, extra=SELECT),
    dict(seed=0, tag="_s13_conv", loss=PLAIN, model=MODERNBERT, max_len=2048, init=STAGE1, extra=SELECT),
    dict(seed=0, tag="_s13bal_conv", loss=PLAIN, model=MODERNBERT, max_len=2048, init=STAGE1, extra=f"{SELECT} --balance_length"),
    # 23/10: sequential stage 1 -> 2 lost the guard skill (probe +6.0 -> +0.28); mix 8k synthetic pairs into every epoch instead
    dict(seed=0, tag="_mix_conv", loss=PLAIN, model=MODERNBERT, max_len=2048, extra=f"{SELECT} --extra_train {SYNTH} --extra_pairs 8000"),
    # 23/10: train on PrimeVul itself (line labels from its own train pairs, temporal split) + the same synthetic mix;
    # SecVulEval -> PrimeVul transfer is capped by the opposite length shortcut of the two datasets
    dict(seed=0, tag="_pv_mix_conv", loss=PLAIN, model=MODERNBERT, max_len=2048, data=PV_LINES,
         extra=f"{SELECT} --extra_train {SYNTH} --extra_pairs 8000"),
    # 23/10: plain _conv retrained on cleaned line labels (label_noise.py). Same recipe as the original _conv run
    # (colab/run_inner_conv.py: lr 3e-5, default epoch selection top5_same_text), only --data differs
    dict(seed=0, tag="_conv_clean", loss=PLAIN, model=MODERNBERT, max_len=2048, data=DATA_CLEAN, extra="--lr 3e-5"),
    # 23/10: no Colab Pro units left -> both sides of the clean-label comparison on a free T4 (no bf16: fp16 + GradScaler).
    # Compare _t4clean with _t4conv only, never with the L4 runs. Run both: --only _t4
    dict(seed=0, tag="_t4conv", loss=PLAIN, model=MODERNBERT, max_len=2048, extra="--lr 3e-5", bf16=False),
    dict(seed=0, tag="_t4clean", loss=PLAIN, model=MODERNBERT, max_len=2048, data=DATA_CLEAN, extra="--lr 3e-5", bf16=False),
]
EXTRA_SEEDS = [
    dict(seed=1, tag="_conv", loss=PLAIN, model=MODERNBERT, max_len=2048),
    dict(seed=1, tag="_contrast03_conv", loss=CONTRAST_LIGHT, model=MODERNBERT, max_len=2048),
    dict(seed=2, tag="_conv", loss=PLAIN, model=MODERNBERT, max_len=2048),
    dict(seed=2, tag="_contrast03_conv", loss=CONTRAST_LIGHT, model=MODERNBERT, max_len=2048),
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
    if any(r.get("data") == DATA_CLEAN for r in RUNS) and not os.path.exists(f"{DATA_CLEAN}/test.jsonl"):
        sh(f"{sys.executable} {D}/prepare_secvuleval.py --out {DATA_CLEAN} --primevul_dir {PRIMEVUL} "
           f"--drop_trivial_flaw --max_flaw_density 0.5")
        sh(f"cmp {DATA}/test.jsonl {DATA_CLEAN}/test.jsonl")   # test must be byte-identical to the unfiltered one
    if any(r.get("data") == SYNTH for r in RUNS) and not os.path.exists(f"{SYNTH}/train.jsonl"):
        if not os.path.exists(f"{PRIMEVUL}/primevul_train.jsonl"):
            qlog("primevul_train.jsonl missing - needed for make_synthetic_guards.py")
            sys.exit(1)
        sh(f"cd {D} && {sys.executable} make_synthetic_guards.py --primevul {PRIMEVUL} --secvul {DATA} --out {SYNTH} --max_train {SYNTH_PAIRS}")
    if any(r.get("data") == PV_LINES for r in RUNS) and not os.path.exists(f"{PV_LINES}/train.jsonl"):
        sh(f"cd {D} && {sys.executable} prepare_primevul_lines.py --primevul {PRIMEVUL} --out {PV_LINES}")


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
    queue = []
    for r in RUNS:
        nm = run_name(r["seed"], r["tag"], r["model"], r["max_len"])
        x = (f"--model {r['model']} --max_len {r['max_len']} {r['loss']} --epochs {r.get('epochs', 4)} "
             f"--patience {r.get('patience', 2)} --data {r.get('data', DATA)} {r.get('extra', '')}")
        if r.get("init"):
            x += f" --init {ROOT}/{r['init']}/ckpt_best.pt"
        common = COMMON if r.get("bf16", True) else COMMON.replace(" --bf16", "")
        queue.append((nm, r["seed"], r["tag"], f"{common} {x}", r.get("init")))
    if a.only:
        queue = [q for q in queue if any(o in q[0] for o in a.only)]
    qlog("queue:", ", ".join(q[0] for q in queue))
    if a.dry:
        for nm, s, t, x, _ in queue:
            print(f"{sys.executable} {D}/train_lines.py {x} --seed {s} --tag {t} --out {ROOT}/{nm}")
            print(f"{sys.executable} {D}/eval_primevul.py --run {ROOT}/{nm} --primevul {PRIMEVUL}")
        return
    import torch  # preinstalled on Colab
    if not torch.cuda.is_available():
        qlog("no GPU visible - start the session with `colab new --gpu L4`; stopping")
        sys.exit(1)
    qlog("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
    if not a.skip_setup:
        setup()
    for nm, s, t, x, init in queue:
        out, log = f"{ROOT}/{nm}", f"{LOGS}/{nm}.log"
        if os.path.exists(f"{out}/primevul_results.json"):
            qlog(f"skip {nm} (trained and scored)")
            continue
        if init and not os.path.exists(f"{ROOT}/{init}/ckpt_best.pt"):
            qlog(f"{nm}: init checkpoint {init}/ckpt_best.pt missing - skipped")
            continue
        t_run = time.time()
        if os.path.exists(f"{out}/results.json"):
            qlog(f"{nm}: results.json exists, PrimeVul scoring only")
        else:
            qlog(f"start {nm}")
            rc = stream(f"{sys.executable} {D}/train_lines.py {x} --seed {s} --tag {t} --out {out}", log)
            if rc != 0 or not os.path.exists(f"{out}/results.json"):
                qlog(f"{nm}: train exit {rc} - moving on (relaunch resumes from ckpt_last.pt)")
                continue
        rc = stream(f"{sys.executable} {D}/eval_primevul.py --run {out} --primevul {PRIMEVUL}", log)
        qlog(f"{nm}: done (PrimeVul exit {rc}) in {(time.time() - t_run) / 3600:.2f} h")
    qlog("ALL_DONE")
    open(f"{LOGS}/ALL_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
