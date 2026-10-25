"""colabctl: run exp5 on Google Colab through the `colab` CLI, from PowerShell on this machine. Every command prints its
output live. Session: -s NAME or $env:COLAB_SESSION (default exp5); GPU for `new`: --gpu (default L4).

  new [--gpu L4]            create the session unless the backend lists it, then setup
  setup                     repo clone, packages, PrimeVul on the VM (idempotent)
  sync                      upload the working copy of experiments/exp5_line_labels/*.py over the VM clone
  presets                   list the named jobs in presets.py
  start PRESET [-- ARGS]    sync, then start a preset as a detached job on the VM
  run NAME -- CMD ...       start any shell command as a detached job (cwd: the exp5 dir on the VM)
  jobs                      job table, GPU, disk
  logs NAME [-f] [-n 40]    print a job's log; -f follows it until the job ends
  kill NAME --yes           stop one job (TERM, then KILL) and confirm it is gone
  mirror PRESET [--down]    follow a job: log, curve, plots, results, ckpt_best of every run it trains -> results/RUN,
                            data/colab_ckpt/RUN, its `pull` dirs when it ends; then any job still running on the VM;
                            keepalive cell every 5 min; --down releases the VM once nothing runs
  push LOCAL REMOTE         upload a file (40 MB chunks above 40 MB, hash checked on the VM)
  pull REMOTE LOCAL         download a file
  exec FILE [--timeout S]   run a local .py in the kernel
  keep [--until JOB] [--down]  keepalive cell every 60 s; stops after --idle min (10) with no job running (vllm and other
                            `service` presets do not count)
  watch [-w NAME ...]       local monitor of every session; exits 10 at the first event (job ended, session gone,
                            kernel not answering, VM idle --idle min): run it in the background to be woken
  status | usage | down --yes

The kernel runs one cell at a time, so every `colab exec` here takes a local lock (work/<session>.lock); uploads and
downloads use the Jupyter contents API and need no lock. Jobs run detached on the VM (vm.py), with their log in
/content/exp5/logs/<job>.log; nothing on the VM is piped into this process.
"""
import argparse
import ctypes
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent                                   # experiments/exp5_line_labels
REPO = E.parent.parent
WORK = HERE / "work"
ROOT = "/content/exp5"
VM_E = "/content/IE105/experiments/exp5_line_labels"
VM_PY = f"{ROOT}/bin/vm.py"
CHUNK = 40 * 1024 * 1024
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
COLAB = shutil.which("colab") or str(Path(sys.executable).parent / "Scripts" / "colab.exe")
S = "exp5"
sys.path.insert(0, str(HERE))


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------- colab CLI
def colab(*args, stdin=None, wall=600, quiet=False):
    """run `colab ARGS`, print its output as it arrives (unless quiet), return (rc, text)"""
    # own process group: a console Ctrl+C reaches only colabctl, which then ends the client itself (finally below)
    p = subprocess.Popen([COLAB, *args], stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV, text=True, encoding="utf-8",
                         errors="replace", bufsize=1, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    timer = threading.Timer(wall, p.kill)
    timer.daemon = True                           # a pending timer must not keep this process alive after Ctrl+C
    timer.start()
    out = []
    try:
        if stdin is not None:
            p.stdin.write(stdin)
            p.stdin.close()
        for line in p.stdout:
            out.append(line)
            if not quiet:
                print(line, end="", flush=True)
            if "Connection was lost" in line:     # the client hangs for minutes after this; end it now, cell() retries
                p.kill()
        rc = p.wait()
    finally:
        timer.cancel()
        if p.poll() is None:                      # Ctrl+C here: end the local CLI client too (the VM is not touched)
            p.kill()
    return rc, "".join(out)


def _pid_alive(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259                                            # STILL_ACTIVE


class Lock:
    """one kernel cell at a time across all colabctl processes; stale when its owner died or its deadline passed"""

    def __init__(self, wall):
        self.path, self.wall = WORK / f"{S}.lock", wall

    def __enter__(self):
        WORK.mkdir(exist_ok=True)
        t0, warned = time.time(), False
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "cmd": " ".join(sys.argv[1:3]),
                                         "deadline": time.time() + self.wall + 60}).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    o = json.loads(self.path.read_text())
                except (OSError, ValueError):
                    o = {}
                if not o or not _pid_alive(o.get("pid", 0)) or time.time() > o.get("deadline", 0):
                    self.path.unlink(missing_ok=True)
                    continue
                if not warned:
                    say(f"waiting for the kernel: '{o.get('cmd')}' (pid {o.get('pid')}) holds the exec lock")
                    warned = True
                if time.time() - t0 > 900:
                    sys.exit("gave up waiting for the exec lock after 15 min")
                time.sleep(2)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def cell(code, timeout=120, quiet=False, tries=3):
    """run python code in the kernel under the lock; the CLI's own --timeout must cover the cell. A lost websocket
    (seen right after uploads) is retried; every vm.py command is safe to repeat (start refuses a running job)."""
    with Lock((timeout + 150) * tries):
        for i in range(tries):
            rc, out = colab("exec", "-s", S, "--timeout", str(timeout), stdin=code, wall=timeout + 120, quiet=quiet)
            if "Connection was lost" not in out:
                return rc, out
            say(f"kernel connection lost (try {i + 1}/{tries}); retrying in 10 s")
            time.sleep(10)
        return rc, out


def vm(*args, timeout=120, quiet=False):
    code = (f"import sys\nsys.argv = {['vm.py', *args]!r}\n"
            f"exec(compile(open({VM_PY!r}).read(), {VM_PY!r}, 'exec'), {{'__name__': '__main__'}})\n")
    return cell(code, timeout, quiet)


def upload(local, remote, tries=3):
    for i in range(tries):
        rc, out = colab("upload", "-s", S, str(local), remote, wall=600, quiet=True)
        if rc == 0 and "rror" not in out:
            return True
        say(f"upload {Path(local).name} failed (try {i + 1}): {out.strip()[-200:]}")
        time.sleep(5)
    return False


def download(remote, local, quiet=True, wall=900):
    """atomic download; False when the remote file does not exist or the transfer failed"""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    rc, out = colab("download", "-s", S, remote, str(part), wall=wall, quiet=True)
    if rc == 0 and part.exists() and part.stat().st_size > 0:
        os.replace(part, local)
        return True
    part.unlink(missing_ok=True)
    if not quiet:
        say(f"download {remote} failed: {out.strip()[-200:]}")
    return False


def session_listed():
    """never guess: a failed listing aborts, because `colab new` on a listed name bills a second VM"""
    rc, out = colab("sessions", wall=90, quiet=True)
    if rc != 0 or ("No active sessions" not in out and "[" not in out):
        sys.exit(f"cannot list sessions (rc {rc}): {out.strip()[-300:]}")
    return f"[{S}]" in out, out


# ----------------------------------------------------------------------------------------------------------- commands
def cmd_new(a):
    listed, out = session_listed()
    print(out.strip())
    if listed:
        say(f"session {S} exists - not creating another VM (that would bill a second one)")
    else:
        say(f"colab new -s {S} --gpu {a.gpu}")
        rc, _ = colab("new", "-s", S, "--gpu", a.gpu, wall=600)
        if rc != 0:
            sys.exit(f"colab new failed (rc {rc})")
        time.sleep(20)                            # the first exec right after `new` can time out while the kernel starts
    cmd_setup(a)


def ensure_vm_helper():
    """make the VM dirs and upload vm.py (both are cheap; called before anything that uses vm.py)"""
    rc, out = cell(f"import os\nfor d in ('{ROOT}/bin', '{ROOT}/upload', '{ROOT}/jobs', '{ROOT}/logs'): os.makedirs(d, exist_ok=True)\n"
                   f"print('dirs ok')\n", 90, quiet=True)
    if "dirs ok" not in out:                      # retry once: a fresh kernel sometimes misses the first reply
        time.sleep(10)
        rc, out = cell(f"import os\nos.makedirs('{ROOT}/bin', exist_ok=True)\nos.makedirs('{ROOT}/upload', exist_ok=True)\nprint('dirs ok')\n", 120)
        if "dirs ok" not in out:
            sys.exit("cannot reach the kernel")
    if not upload(HERE / "vm.py", VM_PY):
        sys.exit("cannot upload vm.py")


def cmd_setup(a):
    ensure_vm_helper()
    rc, out = vm("setup", timeout=1800)
    if "SETUP_DONE" not in out:
        sys.exit("setup did not finish (see the output above)")
    cmd_sync(a)


def cmd_sync(a):
    ensure_vm_helper()
    files = sorted(p for p in E.glob("*.py"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
    WORK.mkdir(exist_ok=True)
    zp = WORK / f"sync_{S}.zip"
    zp.write_bytes(buf.getvalue())
    if not upload(zp, f"{ROOT}/upload/sync.zip"):
        sys.exit("sync upload failed")
    rc, out = vm("unpack", f"{ROOT}/upload/sync.zip", VM_E, timeout=120)
    if "UNPACKED" not in out:
        sys.exit("sync unpack failed")
    say(f"synced {len(files)} files from {E.relative_to(REPO)}")


def cmd_presets(a):
    from presets import PRESETS
    for n, p in PRESETS.items():
        print(f"  {n:16s} {p['note']}" + (f"  [run: {p['run']}]" if p["run"] else ""))


def start_job(name, script):
    WORK.mkdir(exist_ok=True)
    local = WORK / f"job_{name}.sh"
    local.write_bytes(script.encode())            # LF line endings: the script runs under bash on the VM
    if not upload(local, f"{ROOT}/upload/job_{name}.sh"):
        sys.exit("cannot upload the job script")
    rc, out = vm("start", name, f"{ROOT}/upload/job_{name}.sh", timeout=120)
    if "JOB_STARTED" in out:
        say(f"follow it with:  colabctl logs {name} -f" + (f"   (or: colabctl mirror {name})" if name else ""))


def cmd_start(a):
    from presets import PRESETS
    if a.preset not in PRESETS:
        sys.exit(f"unknown preset {a.preset}; see `colabctl presets`")
    p = PRESETS[a.preset]
    steps = list(p["steps"])
    if a.args:
        if not p["takes_args"]:
            sys.exit(f"preset {a.preset} takes no extra arguments")
        steps[-1] += " " + " ".join(shlex.quote(x) for x in a.args)
    if not a.no_sync:
        cmd_sync(a)
    start_job(a.preset, "set -eo pipefail\n" + "\n".join(steps) + "\n")


def cmd_run(a):
    if not a.command:
        sys.exit("usage: colabctl run NAME -- CMD ...")
    if not a.no_sync:
        cmd_sync(a)
    start_job(a.name, "set -eo pipefail\n" + " ".join(a.command) + "\n")


def cmd_jobs(a):
    vm("jobs", timeout=60)


def job_exit(name):
    tmp = WORK / f"exit_{S}_{name}"
    return tmp.read_text().strip() if download(f"{ROOT}/jobs/{name}/exit", tmp) else None


def cmd_logs(a):
    local = WORK / "logs" / f"{S}_{a.name}.log"
    if not download(f"{ROOT}/logs/{a.name}.log", local, quiet=False):
        sys.exit(f"no log for job {a.name} yet")
    text = local.read_text("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in text.split("\n") if l.strip()]
    print("\n".join(lines[-a.n:]), flush=True)
    if not a.f:
        return
    seen = local.stat().st_size
    say(f"following {a.name} every {a.interval} s (Ctrl+C stops following; the job keeps running)")
    while True:
        time.sleep(a.interval)
        if download(f"{ROOT}/logs/{a.name}.log", local):
            data = local.read_bytes()
            if len(data) < seen:
                seen = 0
            new = data[seen:].decode("utf-8", "replace").replace("\r", "\n")
            seen = len(data)
            for l in new.split("\n"):
                if l.strip():
                    print(l, flush=True)
        ext = job_exit(a.name)
        if ext is not None:
            say(f"job {a.name} ended: {'ok' if ext == '0' else ext}")
            return


def cmd_kill(a):
    if not a.yes:
        sys.exit(f"this stops job {a.name} on session {S}; run again with --yes to do it")
    vm("kill", a.name, timeout=90)


# ------------------------------------------------------------------------------------------ job state and safeguards
# 25/10 night: one VM was pruned because nothing touched its kernel after the mirrored job ended (the queued job died
# unsaved), and another was billed 3.4 h by a `keep` loop after its job had ended. So: mirror follows every job on the
# VM, keep stops when no work is left, and `watch` exits on the first event so whoever started it gets woken.
def vm_state(tries=2):
    """the VM's job table (vm.py state) as a dict, or None when the kernel did not answer; also a keepalive"""
    for i in range(tries):
        rc, out = vm("state", timeout=60, quiet=True)
        for l in out.splitlines():
            if l.startswith("STATE_JSON "):
                return json.loads(l[len("STATE_JSON "):])
        if i + 1 < tries:
            time.sleep(10)
    return None


def services():
    from presets import PRESETS
    return {n for n, p in PRESETS.items() if p["service"]}


def busy(st):
    """running jobs that are work; a service (vllm) waiting for clients is not"""
    return [j["name"] for j in st["jobs"] if j["state"] == "running" and j["name"] not in services()]


def describe(st):
    run = [j["name"] for j in st["jobs"] if j["state"] == "running"]
    return f"{st['utc']} UTC | running: {', '.join(run) or 'none'} | GPU {st['gpu'] or '-'}"


def list_sessions():
    """names of the sessions the backend lists, or None when the listing failed"""
    rc, out = colab("sessions", wall=90, quiet=True)
    if rc != 0 or ("No active sessions" not in out and "[" not in out):
        return None
    return set(re.findall(r"\[([\w.-]+)\]", out)) - {"colab"}


def last_termination(name):
    """'pruned at ...' / 'user_requested at ...' from the CLI history of a session. A `colab stop` is followed ~1 s later
    by a second record 'pruned' when another client lists the sessions (seen 25/10 in the selftest), so a 'pruned' within
    10 s of a 'user_requested' counts as user_requested."""
    p = Path.home() / ".config" / "colab-cli" / "history" / f"{name}.jsonl"
    try:
        lines = p.read_text("utf-8", "replace").splitlines()
    except OSError:
        return "no CLI history"
    ends = [json.loads(l) for l in lines if '"session_terminated"' in l]
    if not ends:
        return "no termination recorded in the CLI history"
    from datetime import datetime
    t = lambda o: datetime.fromisoformat(o["timestamp"]).timestamp()
    last = ends[-1]
    near = [o["reason"] for o in ends if abs(t(o) - t(last)) <= 10]
    reason = "user_requested" if "user_requested" in near else last.get("reason")
    return f"{reason} at {last['timestamp'][:19]} UTC"


def lost_or_unreachable():
    """after the kernel did not answer: exit 3 when the backend no longer lists the session, else return"""
    sess = list_sessions()
    if sess is not None and S not in sess:
        say(f"SESSION_GONE {S}: {last_termination(S)}")
        sys.exit(3)
    say(f"kernel of {S} did not answer (session still listed)" if sess is not None else "cannot list sessions")


def release(why):
    say(f"releasing session {S}: {why}")
    colab("stop", "-s", S, wall=180)
    sess = list_sessions()
    say(f"session {S} " + ("still listed - check `colabctl status`" if sess is None or S in sess else "released"))


def is_scratch(name):
    return name.startswith(("smoke", "selftest"))


def local_dirs(run):
    """results/RUN and data/colab_ckpt/RUN; smoke and selftest output goes under work/ so it never mixes with results"""
    if is_scratch(run):
        return WORK / "smoke" / run, WORK / "smoke" / run / "ckpt"
    return E / "results" / run, REPO / "data" / "colab_ckpt" / run


def job_targets(job):
    """(training runs, output dirs) a job writes: the --out of its train_lines.py steps, plus the preset's `pull`"""
    from presets import PRESETS
    if job not in PRESETS:
        return [], []
    p, runs = PRESETS[job], []
    for step in p["steps"]:
        if "train_lines.py" in step:
            for r in re.findall(rf"--out {ROOT}/([\w.-]+)", step):
                if r not in runs:
                    runs.append(r)
    return runs or ([p["run"]] if p["run"] else []), p["pull"]


def cmd_mirror(a):
    job = a.job or a.target
    runs, pulls = job_targets(job)
    if a.run:                            # a preset that trains several runs: mirror only the one named here
        runs = [a.run]
    elif a.job and a.target != a.job:
        runs = [a.target]                # old form: mirror RUN --job JOB
    while True:
        ext = follow(job, runs, pulls, a)
        st = vm_state()
        if st is None:
            lost_or_unreachable()
            sys.exit(4)
        left = busy(st)
        if left and not a.no_follow:     # a queued job took over: keep the kernel busy and mirror that one now
            job = left[0]
            runs, pulls = job_targets(job)
            say(f"job ended ({ext}); still running on {S}: {', '.join(left)} - now following {job}")
            continue
        say(f"NOTHING_RUNNING on {S}: {describe(st)}")
        if a.down:
            release("--down given, all jobs ended and their results are pulled")
        else:
            say(f"the VM is still billed: `colabctl -s {S} down --yes`, or it is pruned after ~30 min of idle kernel")
        return


def follow(job, runs, pulls, a):
    """mirror one job until it ends, then a final pass and the output dirs; returns its exit"""
    say(f"mirror job {job} on {S} every {a.interval} s: runs {runs or '-'}, output dirs {pulls or '-'}")
    say(f"keepalive cell every {a.ping} s; ckpt_last pulled every {a.ckpt_last_every} s (0 = never)")
    st = {"n_best": None, "done": set()}
    t_ping = t_last = time.time()
    while True:
        ext = job_exit(job)              # read before pulling, so the pass after the job ends is a complete final one
        # an idle kernel (no exec for ~29 min) is pruned, file transfers do not count: a short cell keeps it busy
        if a.ping and time.time() - t_ping >= a.ping:
            s = vm_state()
            if s is None:
                lost_or_unreachable()
            else:
                say("ping:", describe(s))
                j = next((x for x in s["jobs"] if x["name"] == job), None)
                if ext is None and (j is None or j["state"] in ("dead", "new")):
                    # never started, or died without writing its exit code (VM restart, OOM kill): nothing will end it
                    ext = f"no exit code (job {j['state'] if j else 'missing'})"
            t_ping = time.time()
        active = mirror_pass(job, runs, st)
        # ckpt_last is only needed to resume on a new VM (each pull is ~300 MB), so it comes rarely
        if active and a.ckpt_last_every and time.time() - t_last >= a.ckpt_last_every:
            C = local_dirs(active)[1]
            if download(f"{ROOT}/{active}/ckpt_last.pt", C / "ckpt_last.pt", wall=1800):
                say(f"pulled ckpt_last.pt of {active} ({(C / 'ckpt_last.pt').stat().st_size / 1e6:.0f} MB)")
            t_last = time.time()
        if ext is not None:
            final_pull(runs)
            for d in pulls:
                local = (WORK / "smoke" if is_scratch(d) else E / "results") / d
                say(f"pulled {pull_tree(f'{ROOT}/{d}', local)} files of {d} -> {local.relative_to(REPO)}")
            say(f"job {job} ended: {'ok' if ext == '0' else ext} - final pass done")
            return ext
        time.sleep(a.interval)


def final_pull(runs):
    """after the job ended: every run's ckpt_best and result files once more, whatever the passes saw. The last best is
    often saved minutes before results.json (when mirror_pass stops pulling a run's checkpoints), and the score files
    are written in place, so a pass may have caught them half-written."""
    for run in runs:
        have = remote_ls(f"{ROOT}/{run}")
        if not have:
            continue
        R, C = local_dirs(run)
        for f in ("ckpt_best.pt", "results.json", "test_scores.jsonl", "primevul_results.json", "primevul_test_scores.jsonl",
                  "curve.json", "train_log.jsonl"):
            if f in have:
                dest = (C if f.endswith(".pt") else R) / f
                ok = download(f"{ROOT}/{run}/{f}", dest, quiet=False, wall=1800)
                say(f"final pull {run}/{f}: " + (f"{dest.stat().st_size / 1e6:.1f} MB" if ok else "FAILED"))


def remote_ls(path):
    """names in a VM directory (sub-directories end in '/'); empty when it does not exist. One call, ~4 s."""
    rc, out = colab("ls", "-s", S, path, wall=120, quiet=True)
    if rc != 0:
        return set()
    return {l.strip() for l in out.splitlines() if l.strip() and not l.startswith("[colab]")}


def pull_tree(remote, local):
    n = 0
    for name in sorted(remote_ls(remote)):
        if name.endswith("/"):
            n += pull_tree(f"{remote}/{name[:-1]}", Path(local) / name[:-1])
        elif download(f"{remote}/{name}", Path(local) / name, quiet=False):
            n += 1
    return n


def mirror_pass(job, runs, st):
    """one pull of everything that exists (each colab call costs ~4 s, so missing files are never requested); returns
    the run being trained now, if any"""
    log = WORK / "logs" / f"{S}_{job}.log"
    if not download(f"{ROOT}/logs/{job}.log", log):
        say("log not reachable (job not started yet, or the session is gone: check `colabctl status`)")
        return None
    text = log.read_text("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in text.split("\n") if l.strip()]
    new = lines[len(st.get("seen", [])):] if st.get("seen") and lines[:len(st["seen"])] == st["seen"] else lines[-5:]
    for l in new[-8:]:                                    # the latest lines since the previous pass
        say("log:", l[:220])
    st["seen"] = lines
    here = remote_ls(ROOT) if runs else set()
    active = None
    for run in runs:
        if run + "/" not in here or run in st["done"]:
            continue
        R, C = local_dirs(run)
        R.mkdir(parents=True, exist_ok=True)
        (R / "train.log").write_text(text, "utf-8")
        have = remote_ls(f"{ROOT}/{run}")
        if "results.json" not in have:
            active = run
        # "saved best" counts over the whole job log; a new one belongs to the run being trained
        b = text.count("saved best")
        if st["n_best"] is None:
            st["n_best"] = b if (C / "ckpt_best.pt").exists() else 0
        if run == active and b > st["n_best"] and "ckpt_best.pt" in have:
            say(f"new best checkpoint (#{b}) of {run}, pulling ckpt_best.pt ...")
            if download(f"{ROOT}/{run}/ckpt_best.pt", C / "ckpt_best.pt", quiet=False, wall=1800):
                say(f"pulled ckpt_best.pt ({(C / 'ckpt_best.pt').stat().st_size / 1e6:.0f} MB)")
                st["n_best"] = b
        for f in ("curve.json", "train_log.jsonl"):
            if f in have:
                download(f"{ROOT}/{run}/{f}", R / f)
        if "plots/" in have:
            for p in sorted(remote_ls(f"{ROOT}/{run}/plots")):
                if p.endswith(".png"):
                    download(f"{ROOT}/{run}/plots/{p}", R / "plots" / p)
        for f in ("results.json", "test_scores.jsonl", "primevul_results.json", "primevul_test_scores.jsonl"):
            if f in have and f not in st.setdefault(run, set()) and download(f"{ROOT}/{run}/{f}", R / f):
                say(f"pulled {run}/{f}")
                st[run].add(f)
        if "primevul_results.json" in st.get(run, set()):
            st["done"].add(run)                           # finished and pulled: later passes skip it
    return active


def cmd_push(a):
    src = Path(a.local)
    size = src.stat().st_size
    if size <= CHUNK:
        ok = upload(src, a.remote)
        say(f"{'uploaded' if ok else 'FAILED'} {src.name} ({size / 1e6:.1f} MB) -> {a.remote}")
        return
    h = hashlib.sha256()
    tmp = WORK / "push"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    with open(src, "rb") as fh:
        i = 0
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
            loc = tmp / f"{src.name}.{i:03d}"
            loc.write_bytes(b)
            rem = f"{ROOT}/upload/{src.name}.{i:03d}"
            if not upload(loc, rem):
                sys.exit(f"chunk {i} failed")
            loc.unlink()
            parts.append(rem)
            say(f"chunk {i + 1} up ({(i + 1) * CHUNK / 1e6:.0f}/{size / 1e6:.0f} MB)")
            i += 1
    vm("join", a.remote, h.hexdigest(), *parts, timeout=600)


def cmd_pull(a):
    ok = download(a.remote, a.local, quiet=False, wall=3600)
    say(f"{'downloaded' if ok else 'FAILED'} {a.remote} -> {a.local}")


def cmd_exec(a):
    cell(Path(a.file).read_text("utf-8"), a.timeout)


def cmd_keep(a):
    say(f"keepalive for {S}: one short cell every {a.interval} s; stops after {a.idle} min with no job running"
        + (f" or once job {a.until} has ended" if a.until else "")
        + ("; then releases the VM (results left on it are lost)" if a.down else ""))
    idle_since, deaf = None, 0
    while True:
        st = vm_state()
        if st is None:
            lost_or_unreachable()
            deaf += 1
            if deaf >= 3:                # a keepalive that cannot reach the kernel only hides the problem
                say(f"KEEP_STOP {S}: kernel did not answer 3 times in a row; the VM is NOT released (results unknown): "
                    f"check `colabctl -s {S} jobs`, then `down --yes`")
                sys.exit(4)
            time.sleep(a.interval)
            continue
        deaf = 0
        say(describe(st))
        work = busy(st)
        if a.until and not any(j["name"] == a.until and j["state"] == "running" for j in st["jobs"]):
            why = f"job {a.until} is not running"
            break
        idle_since = None if work else (idle_since or time.time())
        if idle_since and time.time() - idle_since >= a.idle * 60:
            why = f"no job running for {a.idle} min"
            break
        time.sleep(a.interval)
    say(f"KEEP_STOP {S}: {why}")
    if a.down:
        release(why)
    else:
        say(f"the VM is still billed: `colabctl -s {S} down --yes`, or it is pruned after ~30 min of idle kernel")


def cmd_watch(a):
    """local monitor over every listed session (or -w NAME ...); exits 10 at the first event so the process that started
    it is woken: a job ended, a session disappeared, a kernel stopped answering, or a VM billed with no work for --idle min"""
    global S
    seen, t0, idle, deaf = {}, {}, {}, {}
    listed_before, fails = None, 0
    say(f"watch every {a.interval} s: job exits, lost sessions, kernels not answering, VMs idle for {a.idle} min")
    while True:
        sess = list_sessions()
        if sess is None:
            fails += 1
            say(f"cannot list sessions ({fails})")
            if fails >= 3:
                events = ["LISTING_FAILED: `colab sessions` failed 3 times in a row"]
                break
            time.sleep(60)
            continue
        fails = 0
        mine = {n for n in sess if not a.sessions or n in a.sessions}
        events = [f"SESSION_GONE {n}: {last_termination(n)}" for n in sorted((listed_before or set()) - sess)
                  if not a.sessions or n in a.sessions]
        if not mine and not events:
            say("no sessions listed: nothing billed, nothing to watch")
            return
        for n in sorted(mine):
            S = n
            st = vm_state()
            if st is None:
                deaf[n] = deaf.get(n, 0) + 1
                if deaf[n] >= 2:
                    events.append(f"KERNEL_UNREACHABLE {n}: no reply twice in a row (session still listed)")
                continue
            deaf[n] = 0
            t0.setdefault(n, st["now"])
            prev = seen.get(n, {})
            for j in st["jobs"]:
                was = prev.get(j["name"])
                if j["state"] not in ("running", "new") and (was == "running" or (was is None and (j["ended"] or 0) > t0[n])):
                    events.append(f"JOB_ENDED {n}/{j['name']}: {j['state']}")
            seen[n] = {j["name"]: j["state"] for j in st["jobs"]}
            work = busy(st)
            if work:
                idle.pop(n, None)
            else:
                idle.setdefault(n, time.time())
                if time.time() - idle[n] >= a.idle * 60:
                    events.append(f"IDLE_BILLING {n}: no job running for {a.idle} min ({describe(st)})")
            say(f"{n}: {describe(st)}")
        if events:
            break
        listed_before = sess
        time.sleep(a.interval)
    WORK.mkdir(exist_ok=True)
    with open(WORK / "watch_events.log", "a", encoding="utf-8") as fh:
        for e in events:
            say("WATCH_EVENT", e)
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {e}\n")
    sys.exit(10)


def cmd_status(a):
    colab("status", "-s", S, wall=90)


def cmd_usage(a):
    colab("usage", wall=90)


def cmd_down(a):
    if not a.yes:
        sys.exit(f"this releases the VM of session {S} (all files on it are lost); run again with --yes to do it")
    colab("stop", "-s", S, wall=180)


def main():
    global S
    ap = argparse.ArgumentParser(prog="colabctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", default=os.environ.get("COLAB_SESSION", "exp5"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("--gpu", default="L4")
    sub.add_parser("setup"); sub.add_parser("sync"); sub.add_parser("presets"); sub.add_parser("jobs")
    p = sub.add_parser("start"); p.add_argument("preset"); p.add_argument("args", nargs=argparse.REMAINDER)
    p.add_argument("--no_sync", action="store_true")
    p = sub.add_parser("run"); p.add_argument("name"); p.add_argument("command", nargs=argparse.REMAINDER)
    p.add_argument("--no_sync", action="store_true")
    p = sub.add_parser("logs"); p.add_argument("name"); p.add_argument("-f", action="store_true")
    p.add_argument("-n", type=int, default=40); p.add_argument("--interval", type=int, default=15)
    p = sub.add_parser("kill"); p.add_argument("name"); p.add_argument("--yes", action="store_true")
    p = sub.add_parser("mirror"); p.add_argument("target"); p.add_argument("--job"); p.add_argument("--run")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--ping", type=int, default=300, help="keepalive cell every this many seconds (0 = off)")
    p.add_argument("--ckpt_last_every", type=int, default=3600)
    p.add_argument("--no_follow", action="store_true", help="stop when this job ends, even if another job runs")
    p.add_argument("--down", action="store_true", help="release the VM once nothing runs and everything is pulled")
    p = sub.add_parser("push"); p.add_argument("local"); p.add_argument("remote")
    p = sub.add_parser("pull"); p.add_argument("remote"); p.add_argument("local")
    p = sub.add_parser("exec"); p.add_argument("file"); p.add_argument("--timeout", type=int, default=600)
    p = sub.add_parser("keep"); p.add_argument("--interval", type=int, default=60)
    p.add_argument("--idle", type=float, default=10, help="stop after this many minutes with no job running")
    p.add_argument("--until", help="stop once this job is no longer running")
    p.add_argument("--down", action="store_true", help="release the VM when keep stops (pull results first!)")
    p = sub.add_parser("watch"); p.add_argument("-w", dest="sessions", nargs="*", help="only these sessions")
    p.add_argument("--interval", type=int, default=300); p.add_argument("--idle", type=float, default=15)
    sub.add_parser("status"); sub.add_parser("usage")
    p = sub.add_parser("down"); p.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    S = a.session
    for stream in (sys.stdout, sys.stderr):     # logs contain progress-bar glyphs; a cp1252 console must not crash on them
        stream.reconfigure(encoding="utf-8", errors="replace")
    # the Terminal panel can send one Ctrl+C right after it types a command; ignore Ctrl+C for the first 5 s
    t_start = time.time()

    def early_sigint(sig, frame):
        if time.time() - t_start < 5:
            say("(ignored a Ctrl+C in the first 5 s; press it again to stop)")
            return
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, early_sigint)
    for k in ("args", "command"):        # REMAINDER also swallows our own flags given before `--`
        v = getattr(a, k, None)
        if isinstance(v, list):
            head, tail = (v[:v.index("--")], v[v.index("--") + 1:]) if "--" in v else ([], v)
            if "--no_sync" in head or (not head and "--" not in v and tail[:1] == ["--no_sync"]):
                a.no_sync = True
                head = [x for x in head if x != "--no_sync"]
                if "--" not in v:
                    tail = [x for x in tail if x != "--no_sync"]
            setattr(a, k, head + tail if "--" not in v else tail)
    try:
        globals()[f"cmd_{a.cmd}"](a)
    except KeyboardInterrupt:
        say("stopped (local only; jobs on the VM keep running)")


if __name__ == "__main__":
    main()
