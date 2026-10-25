"""VM-side helper for colabctl.py. It runs inside the Colab kernel through `colab exec` (colabctl sets sys.argv and
execs this file), so every command must return quickly except `setup`.

  setup                    clone the repo, install packages, download PrimeVul (idempotent, output streamed)
  start NAME CMDFILE       start the bash script CMDFILE as a detached job: own process group, no stdin, output
                           appended to /content/exp5/logs/NAME.log, exit code written to /content/exp5/jobs/NAME/exit
  jobs                     one line per job + GPU + disk
  kill NAME                TERM the job's process group, KILL after 15 s, confirm nothing is left
  unpack ZIP DEST          extract an uploaded zip over DEST
  join OUT SHA256 PART...  concatenate uploaded chunks into OUT and check the hash
  ping                     short status line (keepalive)
  state                    STATE_JSON line: every job with state and end time, GPU, VM clock (watch/keep/mirror)

Jobs survive the kernel cell that started them and the CLI connection; a job is never piped into a process that may die.
"""
import glob
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = "/content/exp5"
REPO = "/content/IE105"
E = f"{REPO}/experiments/exp5_line_labels"
JOBS, LOGS, PV = f"{ROOT}/jobs", f"{ROOT}/logs", f"{ROOT}/primevul"
REPO_URL = "https://github.com/phattnt/IE105.git"
PV_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"
PV_FILES = [f"primevul_{s}.jsonl" for s in ("train", "valid", "test", "train_paired", "valid_paired", "test_paired")]
PIP = ["transformers>=4.48", "gdown", "scikit-learn", "matplotlib", "datasets"]
T0 = time.time()


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *a, flush=True)


def sh(cmd):
    """run a command and print its output line by line while it runs (the cell output streams to the local terminal)"""
    say("$", cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print("    " + line.rstrip(), flush=True)
    return p.wait()


def setup():
    for d in (ROOT, JOBS, LOGS, PV, f"{ROOT}/upload"):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(f"{REPO}/.git"):
        sh(f"git clone -q {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh(f"{sys.executable} -m pip install -q --progress-bar off " + " ".join(shlex.quote(p) for p in PIP))
    sh(f"{sys.executable} -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__)'")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'no GPU'")
    if not all(os.path.exists(f"{PV}/{f}") for f in PV_FILES):
        sh(f"gdown -q --folder https://drive.google.com/drive/folders/{PV_FOLDER} -O {PV}")
        for p in glob.glob(f"{PV}/**/primevul_*.jsonl", recursive=True):   # flatten a nested folder
            if os.path.dirname(p) != PV:
                shutil.move(p, f"{PV}/{os.path.basename(p)}")
    have = {f: os.path.getsize(f"{PV}/{f}") // 1_000_000 for f in PV_FILES if os.path.exists(f"{PV}/{f}")}
    say("primevul MB:", have)
    missing = [f for f in PV_FILES if f not in have]
    say("SETUP_DONE" if not missing else f"SETUP_MISSING {missing}")


def _alive(pid):
    try:
        os.waitpid(pid, os.WNOHANG)          # reap it if it is our finished child (otherwise it stays a zombie)
    except ChildProcessError:
        pass
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def state(name):
    d = f"{JOBS}/{name}"
    pid = int(open(f"{d}/pid").read()) if os.path.exists(f"{d}/pid") else None
    ext = open(f"{d}/exit").read().strip() if os.path.exists(f"{d}/exit") else None
    started = float(open(f"{d}/started").read()) if os.path.exists(f"{d}/started") else None
    if ext is not None:
        st = "done" if ext == "0" else f"exit {ext}"
    elif pid and _alive(pid):
        st = "running"
    else:
        st = "dead" if pid else "new"
    return {"name": name, "state": st, "pid": pid, "started": started}


def start(name, cmdfile):
    d = f"{JOBS}/{name}"
    os.makedirs(d, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    s = state(name)
    if s["state"] == "running":
        say(f"JOB_BUSY {name} pid {s['pid']} - not started again")
        return
    for f in ("exit", "pid", "started"):
        if os.path.exists(f"{d}/{f}"):
            os.remove(f"{d}/{f}")
    shutil.copy(cmdfile, f"{d}/cmd.sh")
    with open(f"{d}/wrap.sh", "w") as fh:
        fh.write(f'echo "== $(date -u "+%F %T") UTC start {name}"\n'
                 f'cd {shlex.quote(E)} && bash {d}/cmd.sh\n'
                 f'rc=$?\necho "== $(date -u "+%F %T") UTC exit $rc"\necho $rc > {d}/exit\n')
    log = f"{LOGS}/{name}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log, "a") as out:
        p = subprocess.Popen(["bash", f"{d}/wrap.sh"], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env)
    open(f"{d}/pid", "w").write(str(p.pid))
    open(f"{d}/started", "w").write(str(time.time()))
    say(f"JOB_STARTED {name} pid {p.pid} log {log}")
    print(open(f"{d}/cmd.sh").read(), flush=True)


def gpu_query(fields):
    """nvidia-smi fields as one csv line; '' on a CPU VM (no nvidia-smi)"""
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def _last_line(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - 4000))
            tail = fh.read().decode("utf-8", "replace").replace("\r", "\n")
        lines = [l for l in tail.split("\n") if l.strip()]
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def jobs():
    names = sorted(os.listdir(JOBS)) if os.path.isdir(JOBS) else []
    if not names:
        say("no jobs")
    for n in names:
        s = state(n)
        age = f"{(time.time() - s['started']) / 3600:5.2f} h" if s["started"] else "   -   "
        size = os.path.getsize(f"{LOGS}/{n}.log") // 1000 if os.path.exists(f"{LOGS}/{n}.log") else 0
        print(f"  {n:28s} {s['state']:9s} pid {s['pid'] or '-':>7} {age} log {size:6d} kB | {_last_line(f'{LOGS}/{n}.log')}", flush=True)
    gpu = gpu_query("name,utilization.gpu,memory.used,memory.total")
    disk = shutil.disk_usage("/content")
    print(f"  GPU: {gpu or 'none'} | disk free {disk.free / 1e9:.0f} GB", flush=True)


def kill(name):
    s = state(name)
    if s["state"] != "running":
        say(f"{name} is not running ({s['state']})")
        return
    pg = s["pid"]
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(pg, sig)
        except ProcessLookupError:
            break
        for _ in range(wait):
            time.sleep(1)
            if not subprocess.run(["pgrep", "-g", str(pg)], capture_output=True, text=True).stdout.split():
                break
    left = subprocess.run(["pgrep", "-g", str(pg)], capture_output=True, text=True).stdout.split()
    _alive(pg)
    if left:
        say(f"STILL_ALIVE {name}: pids {left}")
    else:
        open(f"{JOBS}/{name}/exit", "w").write("killed")
        say(f"KILLED {name} (process group {pg}), nothing left")


def unpack(zpath, dest):
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
        say(f"UNPACKED {len(z.namelist())} files into {dest}")


def join(out, sha, parts):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h = hashlib.sha256()
    with open(out + ".part", "wb") as fo:
        for p in parts:
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
                    fo.write(b)
    if h.hexdigest() != sha:
        say(f"JOIN_BAD_HASH {out}: {h.hexdigest()} != {sha}")
        return
    os.replace(out + ".part", out)
    for p in parts:
        os.remove(p)
    say(f"JOINED {out} ({os.path.getsize(out) / 1e6:.1f} MB, sha256 ok)")


def ping():
    run = [n for n in (sorted(os.listdir(JOBS)) if os.path.isdir(JOBS) else []) if state(n)["state"] == "running"]
    gpu = gpu_query("utilization.gpu,memory.used")
    print(f"{time.strftime('%H:%M:%S')} UTC | running: {', '.join(run) or 'none'} | GPU {gpu or '-'}", flush=True)


def state_json():
    """every job with its state and end time (VM clock), one line for colabctl to parse; also a keepalive"""
    out = []
    for n in sorted(os.listdir(JOBS)) if os.path.isdir(JOBS) else []:
        s = state(n)
        ef = f"{JOBS}/{n}/exit"
        s["ended"] = os.path.getmtime(ef) if os.path.exists(ef) else None
        out.append(s)
    gpu = gpu_query("utilization.gpu,memory.used")
    print("STATE_JSON " + json.dumps({"jobs": out, "gpu": gpu, "now": time.time(), "utc": time.strftime("%H:%M:%S")}),
          flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"setup": setup, "start": start, "jobs": jobs, "kill": kill, "unpack": unpack, "ping": ping, "state": state_json,
     "join": lambda out, sha, *parts: join(out, sha, list(parts))}[cmd](*args)
