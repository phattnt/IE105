# Colab Pro runner for exp5 (Colab CLI, L4)

These are the scripts that ran the converged `conv2` (fix-contrastive) run on 2026-10-21. The `conv` run (plain line labels) used the same scripts with `conv2` replaced by `conv` and `run_inner_conv.py` in place of `run_inner_conv2.py`.

Prerequisite: the Colab CLI is installed at `~/.local/bin/colab` and logged in. The token is stored at `~/.config/colab-cli/token.json`. It was deleted after the run, so you must log in again.

## Files
| File | Runs where | What it does |
|---|---|---|
| `setup.py` | VM | Clones the repo, installs gdown and datasets, and runs `prepare_secvuleval.py`, which builds the SecVulEval line data and downloads PrimeVul. |
| `run_inner_conv*.py` | VM | Runs `train_lines.py`, then `eval_primevul.py`. It resumes from `ckpt_last.pt` and skips any step that is already finished. |
| `run.py` | VM | Starts `run_inner` detached, so the kernel stays free for keepalive cells. |
| `keep.py` / `keep.sh` | VM / local | Keepalive (see note below). |
| `poll.sh` | local | Every 2 min it mirrors logs and pulls `ckpt_last` and `ckpt_best`. When the VM is gone it starts `relaunch.sh`. When everything is done it downloads the results into `results/…` and stops the VM. |
| `relaunch.sh` | local | Creates a new L4 session, runs `setup.py`, uploads the checkpoints in 40 MB chunks (single uploads over ~100 MB fail), then runs `run.py`. |

## Keepalive
Colab Pro has no background execution. The VM was reclaimed about 20 minutes after the last client connection closed, and a `colab exec` connection drops after about 20 minutes. Without `keep.sh` running the whole time, each VM lived only about 40 minutes.

## Usage
1. Copy the scripts into a work dir.
2. Rename the matching `run_inner_conv*.py` to `run_inner.py`.
3. Set `S=<work dir>`.
4. Run `relaunch.sh` once, then start `poll.sh` and `keep.sh` with `nohup`.

The session name (`conv2`) is hard-coded in the scripts.

Throughput on L4 (ModernBERT-base, 2048 tokens, bf16):
- about 0.5 s per step, about 30 min per epoch, plus about 5 min for validation
- the full PrimeVul eval takes about 15 min
- the cost is 1.54 compute units per hour per L4
