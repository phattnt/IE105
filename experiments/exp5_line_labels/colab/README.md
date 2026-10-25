# Colab runner for exp5 (`colabctl`)

Runs exp5 on Google Colab Pro from PowerShell on Windows through the `colab` CLI (0.7.5). Replaces the bash scripts and
`colab_exp5.py` queue used until 24/10 (archived in `archive/colab_old_2026-10-24.zip`, git-ignored).

| File | Runs where | What it does |
|---|---|---|
| `colabctl.py` | local | The controller: session, setup, sync, jobs, logs, mirror, push/pull, keepalive, watch. All output live. |
| `vm.py` | VM (in the kernel) | Setup, detached jobs (`/content/exp5/jobs/<name>/`, log `/content/exp5/logs/<name>.log`), kill, unpack, join chunks, job state as JSON. |
| `presets.py` | local | Named jobs: data prep, smoke, seed replications, PrimeVul runs, vLLM + llm_audit. |

## Usage (PowerShell, in this folder)
```powershell
$py = 'C:\Users\kakushi\AppData\Local\Programs\Python\Python312\python.exe'   # `python` in PowerShell is the msys one
& $py -u colabctl.py new --gpu L4          # VM + repo + packages + PrimeVul + sync (~2 min)
& $py -u colabctl.py presets               # list named jobs
& $py -u colabctl.py start smoke           # sync the working copy, start the job detached
& $py -u colabctl.py mirror smoke          # follow it: log, curve, plots, results, ckpt_best; keepalive every 5 min
& $py -u colabctl.py jobs                  # job table + GPU + disk
& $py -u colabctl.py logs smoke -f         # raw log, follow until the job ends
& $py -u colabctl.py kill smoke --yes      # stop one job (TERM, then KILL), confirmed
& $py -u colabctl.py down --yes            # release the VM (all VM files are lost)
```
Other session: `-s exp5llm` (use a separate VM for vLLM: installing it upgrades torch). Another GPU: `new --gpu A100`.
Mirror targets: `results/<run>/` (train.log, curve.json, train_log.jsonl, plots, results) and `data/colab_ckpt/<run>/`
(ckpt_best on every new best, ckpt_last once an hour) for every run the preset trains, plus the preset's `pull` dirs
(LLM outputs) into `results/<dir>/` when the job ends. Smoke and selftest output goes under `work/smoke/`.

## Safeguards (since 25/10): every billed VM has a client tied to its work
```powershell
& $py -u colabctl.py mirror base --down               # follows base, then any job still running (a queued one), pulls
                                                      # everything, then releases the VM once nothing runs
& $py -u colabctl.py -s exp5llm keep --down           # LLM VM: keepalive until no job but vllm is left for 10 min, then release
& $py -u colabctl.py watch                            # in the background: exits 10 at the first event, see below
```
* `mirror` no longer stops when its job ends: if another job is running on the VM it follows that one (keepalive
  included), so a queued job is never left without a client. When nothing runs it prints `NOTHING_RUNNING`; with
  `--down` it releases the VM, otherwise it says the VM is still billed.
* `keep` stops after `--idle` minutes (10) with no job running, or when `--until JOB` is no longer running. A preset
  marked `service` (vllm) does not count as work, so a vLLM server with no client left is idle.
  `keep --down` releases the VM without pulling anything: use it only when results are pulled by `mirror` or by hand.
* `watch` (all listed sessions, or `-w NAME ...`) polls every 5 min and exits with code 10 at the first of:
  `JOB_ENDED session/job`, `SESSION_GONE` (with the reason from the CLI history: `pruned` or `user_requested`),
  `KERNEL_UNREACHABLE` (no reply twice), `IDLE_BILLING` (no work for `--idle` min, 15). Events also go to
  `work/watch_events.log`. Claude runs it as a background command, so the exit wakes it to pull results, start the
  next job or release the VM; re-run it after each event. With no session listed it exits 0 at once.
* `presets selftest` / `selftest_after`: two 2-minute jobs, the second queued behind the first, for checking all three on
  a CPU VM (`colab new -s exp5test`, no `--gpu`).

## What we learned (24–25/10)
* **Colab never reclaimed a busy VM.** The CLI logs `session_terminated` with reason `user_requested` only for
  `colab stop` and `pruned` when the backend dropped the VM. Through 24/10 all 12 terminations were `user_requested`;
  every lost run was ours: a chained `pkill` (23/10 16:50 UTC), scripts that called `colab stop`/relaunch, or stopping
  the VM by hand.
* **25/10 night, the two failures the safeguards above fix.** (1) An A100 VM whose kernel got no cell for ~29 min was
  `pruned` (03:53), and later the training VM was pruned after its mirror ended with the `base` job, killing the queued
  `s13b` job unsaved. (2) A `keep` loop on the A100 ran 3.4 h after its LLM job had ended (~18 units for nothing),
  because nothing told anyone the job had ended.
* **`Connection was lost`** is the CLI client's websocket to the kernel dropping (75 `WebSocketClosedError` in the VM's
  Jupyter log over 7.5 h on 23/10, while the old poller pulled a 300 MB checkpoint every 3 min). It never touched the
  detached trainer. The client then hangs for minutes; `colabctl` kills it at once and retries the cell (up to 3 times;
  every `vm.py` command is safe to repeat).
* Limits (Colab FAQ): paid runtimes run up to 24 h; idle runtimes are disconnected after an unpublished time; background
  execution only on higher plans. Hence `mirror` sends a short kernel cell every 5 min, and a queue per VM must fit in 24 h
  (`train_lines.py` resumes from `ckpt_last.pt` on a new VM: `push` it back to `/content/exp5/<run>/ckpt_last.pt`).
* The app's Terminal panel can send a Ctrl+C right after typing a command; `colabctl` ignores Ctrl+C in its first 5 s, and
  the `colab` child processes run in their own process group. Do not wrap `colabctl` in a `.cmd` (Ctrl+C then stops at
  "Terminate batch job (Y/N)?").
* Never `colab new` a name the backend still lists (a second billed VM); `colabctl new` checks and aborts if it cannot list.
* `colab upload` fails above ~100 MB: `push` sends 40 MB chunks and checks the sha256 on the VM.
