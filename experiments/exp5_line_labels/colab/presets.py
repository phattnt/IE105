"""Named jobs for `colabctl start <preset>`. Each preset is a bash script run on the VM from the exp5 directory of the
synced repo clone (set -eo pipefail: the first failing step ends the job). Data preparation is part of every preset that
needs it and is skipped when the output already exists, so a preset can simply be started again after a VM reset;
train_lines.py resumes from <run>/ckpt_last.pt by itself.

A training preset has `run` = its output directory under /content/exp5 (= results/<run> locally), used by `mirror`.
Run names follow the old scheme slide_{model}_{max_len}_s{seed}{tag} so they line up with results/.
"""
ROOT = "/content/exp5"
PV = f"{ROOT}/primevul"
SECVUL = f"{ROOT}/secvul_lines"
SECVUL_CLEAN = f"{ROOT}/secvul_lines_clean"
SYNTH = f"{ROOT}/synth_guards"
PV_LINES = f"{ROOT}/primevul_lines"
PV_B80K = f"{ROOT}/primevul_lines_b80k"

MODERNBERT = "answerdotai/ModernBERT-base"
COMMON = "--strategy slide --model answerdotai/ModernBERT-base --max_len 2048 --train_tokens 8192 --bf16 --ckpt_every 400"
PLAIN = "--lam_line 0 --lam_fn 0"
CONTRAST = "--lam_line 1.0 --lam_fn 1.0 --margin 1"
CONTRAST_LIGHT = "--lam_line 0.3 --lam_fn 1.0 --margin 1"
DEBIAS = "--select top5_pair_debias"

PREP = {
    SECVUL: f"python prepare_secvuleval.py --out {SECVUL} --primevul_dir {PV}",
    SECVUL_CLEAN: f"python prepare_secvuleval.py --out {SECVUL_CLEAN} --primevul_dir {PV} --drop_trivial_flaw --max_flaw_density 0.5",
    PV_LINES: f"python prepare_primevul_lines.py --primevul {PV} --out {PV_LINES}",
    PV_B80K: f"python prepare_primevul_lines.py --primevul {PV} --out {PV_B80K} --extra_benign 80000 --extra_benign_valid 3000 --seed 0",
    SYNTH: f"python make_synthetic_guards.py --primevul {PV} --secvul {SECVUL} --out {SYNTH} --max_train 40000",
}
NEEDS = {SYNTH: [SECVUL]}


def prep(*dirs):
    """idempotent data steps for the given dirs (and what they depend on), in order"""
    out, seen = [], set()

    def add(d):
        for dep in NEEDS.get(d, []):
            add(dep)
        if d not in seen:
            seen.add(d)
            out.append(f'if [ -f {d}/train.jsonl ]; then echo "data ok: {d}"; else {PREP[d]}; fi')
    for d in dirs:
        add(d)
    return out


def run_name(seed, tag):
    return f"slide_ModernBERT-base_2048_s{seed}{tag}"


def train(seed, tag, data, args, init=None, epochs=4, patience=2):
    """train_lines.py, then eval_primevul.py; each step is skipped when its result file already exists"""
    run = run_name(seed, tag)
    out = f"{ROOT}/{run}"
    x = f"{COMMON} {args} --data {data} --epochs {epochs} --patience {patience} --seed {seed} --tag {tag} --out {out}"
    if init:
        x += f" --init {ROOT}/{init}/ckpt_best.pt"
    return [
        f'if [ -f {out}/results.json ]; then echo "trained already: {run}"; else python train_lines.py {x}; fi',
        f'if [ -f {out}/primevul_results.json ]; then echo "scored already: {run}"; else python eval_primevul.py --run {out} --primevul {PV}; fi',
    ]


PRESETS = {}


def preset(name, steps, run=None, note="", takes_args=False, service=False, pull=()):
    """takes_args: `colabctl start NAME -- ARGS` appends ARGS to the last step. service: a server that runs until killed
    (vllm); keep/watch/mirror do not count it as work, so a VM with only a service left is reported idle. pull: output
    dirs under /content/exp5 that `mirror` copies to results/ when the job ends (training runs are found by itself)."""
    PRESETS[name] = {"steps": steps, "run": run, "note": note, "takes_args": takes_args, "service": service,
                     "pull": list(pull)}


# data only
preset("prep_secvul", prep(SECVUL), note="SecVulEval line labels")
preset("prep_pv_b80k", prep(PV_B80K), note="PrimeVul pairs + 80k benign (the pv1 data)")
preset("prep_synth", prep(SYNTH), note="40k synthetic 'guard removed' pairs (needs SecVulEval)")

# smoke test: 40 functions, 1 epoch, its own output dir; checks GPU, data and the training code end to end
preset("smoke", prep(PV_B80K) + [
    f"python train_lines.py {COMMON} {PLAIN} --data {PV_B80K} --epochs 1 --patience 0 --limit 40 --seed 0 --tag _smoke "
    f"--out {ROOT}/smoke --eval_every 0.5 --log_every 5",
], run="smoke", note="tiny end-to-end training check (~5 min incl. data)")

# seed replication of the two main line-localisation results (roadmap phase 1). Recipes copied from results/*/results.json:
# s0_conv used lr 3e-5 and the default epoch selection; synth1 -> s13_conv used lr 2e-5 and top5_pair_debias.
for s in (1, 2):
    preset(f"conv_s{s}", prep(SECVUL) + train(s, "_conv", SECVUL, f"{PLAIN} --lr 3e-5"),
           run=run_name(s, "_conv"), note=f"plain SecVulEval baseline, seed {s} (~2 h L4)")
    preset(f"s13_s{s}", prep(SYNTH) + train(s, "_synth1", SYNTH, f"{CONTRAST} {DEBIAS}", epochs=1, patience=0)
           + train(s, "_s13_conv", SECVUL, f"{PLAIN} {DEBIAS}", init=run_name(s, "_synth1")),
           run=run_name(s, "_s13_conv"), note=f"synthetic stage 1 then SecVulEval stage 2, seed {s} (~3 h L4)")
    preset(f"contrast03_s{s}", prep(SECVUL) + train(s, "_contrast03_conv", SECVUL, CONTRAST_LIGHT),
           run=run_name(s, "_contrast03_conv"), note=f"light pair contrast, seed {s} (~2 h L4)")

# 25/10 new baseline (recipe audit): the s0_conv recipe, but one epoch-selection rule for every line-level comparison
# (valid Top-5 + Top-10), and a 2-epoch schedule so the LR decays where valid peaks (it always peaked at epoch 1-2 with the
# LR still near its maximum), validated 4x per epoch to catch the peak. Seeds 0 and 1 run one after the other in one job.
BASE = f"{PLAIN} --lr 3e-5 --select top5_top10 --eval_every 0.25"
for s in (0, 1):
    preset(f"base_s{s}", prep(SECVUL) + train(s, "_base2", SECVUL, BASE, epochs=2, patience=0),
           run=run_name(s, "_base2"), note=f"new baseline, seed {s} (~2 h L4)")
preset("base", prep(SECVUL) + train(0, "_base2", SECVUL, BASE, epochs=2, patience=0)
       + train(1, "_base2", SECVUL, BASE, epochs=2, patience=0),
       run=run_name(0, "_base2"), note="new baseline, seeds 0 then 1 (~4 h L4); mirror seed 1 with --run")

# 25/10 listwise loss (Notion task "Thử loss xếp hạng theo hàm..."): the base2 recipe with only the loss changed, one seed
# (compare with both base2 seeds). smoke_lw checks the new code path end to end on 40 SecVulEval functions.
# --mask_contradict was dropped after the pre-run debate: 41% of the lines it masks are trivial ({, }, break;), ~40% match
# a clean line by accident, and the fixed version's 0 is the correct label; it would also confound the loss change.
LW = f"{BASE} --loss listwise --lam_bce 0.3"
preset("smoke_lw", prep(SECVUL) + [
    f"python train_lines.py {COMMON} {LW} --data {SECVUL} --epochs 1 --patience 0 --limit 40 --seed 0 --tag _smoke_lw "
    f"--out {ROOT}/smoke_lw --eval_every 0.5 --log_every 5",
    f"python eval_primevul.py --run {ROOT}/smoke_lw --primevul {PV} --limit 20",
], run="smoke_lw", note="tiny check of the listwise loss incl. PrimeVul scoring and mirror's final pull (~5 min)")
preset("lw_s0", prep(SECVUL) + train(0, "_lw", SECVUL, LW, epochs=2, patience=0),
       run=run_name(0, "_lw"), note="base2 recipe with the listwise loss + contradiction mask, seed 0 (~2 h L4)")

# synthetic stage 1 (the synth1 recipe unchanged) then the new-baseline recipe as stage 2: the only difference from
# base_s{s} is --init, so base2 vs s13b2 isolates the synthetic pre-training. `after_base` waits for the base job.
# 25/10: started once, lost with the VM after ~1 h, then dropped (expected gain within the seed-to-seed spread).
S1 = f"{CONTRAST} {DEBIAS} --lr 2e-5"
AFTER_BASE = [f"while [ ! -f {ROOT}/jobs/base/exit ]; do sleep 60; done; echo \"base exit: $(cat {ROOT}/jobs/base/exit)\""]
s13b = []
for s in (0, 1):
    s13b += train(s, "_synth1", SYNTH, S1, epochs=1, patience=0) + train(s, "_s13b2", SECVUL, BASE, init=run_name(s, "_synth1"),
                                                                          epochs=2, patience=0)
preset("s13b", prep(SYNTH) + s13b, run=run_name(0, "_s13b2"),
       note="synthetic stage 1 + new-baseline stage 2, seeds 0 then 1 (~5 h L4); mirror the others with --run")
preset("s13b_after_base", AFTER_BASE + prep(SYNTH) + s13b, run=run_name(0, "_s13b2"),
       note="s13b, started once the base job has exited")

# cleaned SecVulEval line labels (LABEL_NOISE.md): the s0_conv recipe with only --data changed
preset("conv_clean", prep(SECVUL, SECVUL_CLEAN) + [f"cmp {SECVUL}/test.jsonl {SECVUL_CLEAN}/test.jsonl"]
       + train(0, "_conv_clean", SECVUL_CLEAN, f"{PLAIN} --lr 3e-5"),
       run=run_name(0, "_conv_clean"), note="plain s0_conv recipe on cleaned line labels (~2 h L4)")

# PrimeVul runs
preset("pv1", prep(PV_B80K) + train(0, "_pv1", PV_B80K, f"{PLAIN} --select top5_top10 --eval_every 0.5 --log_every 50",
                                    epochs=10, patience=3),
       run=run_name(0, "_pv1"), note="PrimeVul-only line BCE (exists: results/slide_..._s0_pv1)")
preset("pv_mix", prep(PV_LINES, SYNTH) + train(0, "_pv_mix_conv", PV_LINES, f"{PLAIN} {DEBIAS} --extra_train {SYNTH} --extra_pairs 8000"),
       run=run_name(0, "_pv_mix_conv"), note="PrimeVul + 8k synthetic pairs mixed into every epoch (never run)")
preset("mix", prep(SECVUL, SYNTH) + train(0, "_mix_conv", SECVUL, f"{PLAIN} {DEBIAS} --extra_train {SYNTH} --extra_pairs 8000"),
       run=run_name(0, "_mix_conv"), note="SecVulEval + 8k synthetic pairs mixed (never run)")

# LLM (use its own session, e.g. -s exp5llm: installing vLLM upgrades torch on that VM). The server is a long-running job;
# stop it with `colabctl kill vllm`. llm_audit takes its arguments after `--`, inputs go up with `colabctl push`.
preset("vllm", [
    "pip install -U --progress-bar off 'vllm>=0.17'",
    "exec python -m vllm.entrypoints.openai.api_server --model ${LLM_MODEL:-Qwen/Qwen3.5-9B} --served-model-name llm "
    "--reasoning-parser qwen3 --max-model-len ${LLM_MAX_LEN:-20480} --gpu-memory-utilization 0.92 "
    "--max-num-seqs ${LLM_MAX_SEQS:-32} --dtype bfloat16 --quantization ${LLM_QUANT:-fp8} --port 8000",
], note="vLLM OpenAI server for Qwen3.5-9B on port 8000 (FP8)", service=True)
preset("llm_audit", ["python llm_audit.py run"], note="llm_audit.py run <args after -->, needs the vllm job up",
       takes_args=True)

# phase 2: cross-function context (build_context.py) on PrimeVul pairs. Arms: audit (no context), audit_ctx (real),
# audit_ctxfake (control). Inputs go up first: colabctl push inputs/ctx_<split>_pairs.jsonl /content/exp5/upload/...
# llm_ctx_valid is the check run (60 valid pairs); llm_ctx_test is the one pre-registered test run.
WAIT_VLLM = "until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do echo 'waiting for vllm'; sleep 30; done"
CTX_ARMS = "--arms audit audit_ctx audit_ctxfake --workers 24 --max_model_len 20480"
for split, extra in (("valid", " --limit 60"), ("test", "")):
    src, out = f"{ROOT}/upload/ctx_{split}_pairs.jsonl", f"{ROOT}/llm_ctx_{split}"
    preset(f"llm_ctx_{split}", [WAIT_VLLM, f"python llm_audit.py run --scores {src} --out {out} {CTX_ARMS}{extra}",
                                f"python llm_audit.py score --scores {src} --out {out}"],
           note=f"context arms on PrimeVul {split} pairs{extra} (needs the vllm job)", pull=[f"llm_ctx_{split}"])

# checks for colabctl itself (no GPU work): two short jobs, the second queued behind the first, each writing a file.
# `mirror selftest` must follow both, pull work/selftest, then report the VM idle; `watch` must report each exit.
TICK = "for i in $(seq 1 {n}); do echo \"{job} tick $i\"; sleep 15; done"
preset("selftest", [f"mkdir -p {ROOT}/selftest", TICK.format(n=8, job="selftest"),
                    f"date -u > {ROOT}/selftest/a.txt"], note="colabctl check: 2 min job", pull=["selftest"])
preset("selftest_after", [f"while [ ! -f {ROOT}/jobs/selftest/exit ]; do sleep 10; done", TICK.format(n=8, job="selftest_after"),
                          f"date -u > {ROOT}/selftest/b.txt"], note="colabctl check: 2 min job queued behind selftest",
       pull=["selftest"])
