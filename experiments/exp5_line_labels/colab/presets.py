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
SECVUL_DIFF = f"{ROOT}/secvul_lines_diff"
SYNTH = f"{ROOT}/synth_guards"
PV_LINES = f"{ROOT}/primevul_lines"
PV_CLEAN = f"{ROOT}/primevul_lines_clean"
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
    SECVUL_DIFF: f"python prepare_secvuleval.py --out {SECVUL_DIFF} --primevul_dir {PV} --diff_clean --drop_trivial_flaw --max_flaw_density 0.5",
    PV_LINES: f"python prepare_primevul_lines.py --primevul {PV} --out {PV_LINES}",
    PV_CLEAN: f"python prepare_primevul_lines.py --primevul {PV} --out {PV_CLEAN} --diff_clean",
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
# Tier 1 theoretically grounded presets: clean diff labels + collar_radius=2 + norm_maxmean + listwise
LW_CLEAN = f"{PLAIN} --lr 3e-5 --loss listwise --lam_bce 0.3 --collar_radius 2 --arch norm_maxmean --select top5_top10 --eval_every 0.25"
preset("pv_lw_clean", prep(PV_CLEAN) + train(0, "_pv_lw_clean", PV_CLEAN, LW_CLEAN, epochs=4, patience=2),
       run=run_name(0, "_pv_lw_clean"),
       note="PrimeVul pairs only (no 80k benigns, ~15x faster/epoch) + diff_clean + collar_radius=2 + norm_maxmean + listwise")
preset("secvul_lw_clean", prep(SECVUL_DIFF) + train(0, "_secvul_lw_clean", SECVUL_DIFF, LW_CLEAN, epochs=2, patience=0),
       run=run_name(0, "_secvul_lw_clean"),
       note="SecVulEval diff_clean + collar_radius=2 + norm_maxmean + listwise (~2 h L4)")
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

# OpenVul 4B (Phase 1): OpenVul-Qwen3-4B-SFT-ep3 with symmetric neutral hints (K=3)
WAIT_VLLM = "until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do echo 'waiting for vllm'; sleep 30; done"
OPENVUL_MODEL = "Leopo1d/OpenVul-Qwen3-4B-SFT-ep3"
preset("vllm_openvul", [
    "pip install -U --progress-bar off 'vllm>=0.17'",
    f"exec python -m vllm.entrypoints.openai.api_server --model {OPENVUL_MODEL} --served-model-name llm "
    "--reasoning-parser qwen3 --max-model-len 16384 --gpu-memory-utilization 0.90 "
    "--max-num-seqs 32 --dtype bfloat16 --port 8000",
], note="vLLM OpenAI server for OpenVul-4B on port 8000 (bfloat16)", service=True)

preset("openvul_smoke", [
    WAIT_VLLM,
    f"python llm_audit.py run --scores {ROOT}/upload/test_scores.jsonl --out {ROOT}/openvul_smoke --arms audit_hint audit --k 3 --hint_style neutral --workers 16 --limit 10 --tokenizer {OPENVUL_MODEL}",
    f"python llm_audit.py score --scores {ROOT}/upload/test_scores.jsonl --out {ROOT}/openvul_smoke",
], note="Smoke test (10 pairs) for OpenVul-4B with symmetric hints K=3", pull=["openvul_smoke"])

preset("openvul_test", [
    WAIT_VLLM,
    f"python llm_audit.py run --scores {ROOT}/upload/test_scores.jsonl --out {ROOT}/openvul_test --arms audit_hint audit plain --k 3 --hint_style neutral --workers 24 --tokenizer {OPENVUL_MODEL}",
    f"python llm_audit.py score --scores {ROOT}/upload/test_scores.jsonl --out {ROOT}/openvul_test",
], note="Full test for OpenVul-4B with symmetric hints K=3 on PrimeVul test pairs", pull=["openvul_test"])

# Plan v2 (Phase 1): OpenVul-Qwen3-4B-GRPO with prefix caching + 50-pair CLEAN smoke gate
OPENVUL_GRPO_MODEL = "Leopo1d/OpenVul-Qwen3-4B-GRPO"
preset("vllm_openvul_grpo", [
    "pip install -U --progress-bar off 'vllm>=0.17'",
    f"exec python -m vllm.entrypoints.openai.api_server --model {OPENVUL_GRPO_MODEL} --served-model-name llm "
    "--reasoning-parser qwen3 --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 "
    "--max-num-seqs 32 --dtype bfloat16 --port 8000",
], note="vLLM OpenAI server for OpenVul-Qwen3-4B-GRPO on port 8000 (bfloat16, prefix-cache)", service=True)

GD1_SANITY_CHECK = (
    "python - << 'PY'\n"
    "import json, sys\n"
    "s = json.load(open('/content/exp5/gd1_sanity/scores/summary.json'))\n"
    "for arm, v in s.items():\n"
    "    m = v.get('modes', {})\n"
    "    tot_m = max(sum(m.values()), 1)\n"
    "    tf = 100.0 * m.get('think_forced', 0) / tot_m\n"
    "    print(f'SANITY {arm}: answer_rate={v[\"answer_rate\"]}% think_forced={tf:.1f}% modes={m} gen_mean={v[\"gen_tokens\"][\"mean\"]}')\n"
    "    if v['answer_rate'] < 90.0 or tf > 25.0:\n"
    "        sys.exit(f'SANITY_GATE_FAIL on {arm}: answer_rate={v[\"answer_rate\"]} think_forced={tf:.1f}%')\n"
    "print('SANITY_GATE_PASS')\n"
    "PY"
)
GD1_SCORES = f"{ROOT}/upload/test_scores.jsonl"
GD1_OUT = f"{ROOT}/gd1_smoke50"
GD1_SUB = f"{ROOT}/upload/smoke50_clean_pairs.json"
GD1_LEAK = f"{ROOT}/upload/leak_audit_openvul_grpo.json"
GD1_TPL = f"{ROOT}/upload/openvul_grpo_plain_templates.json"
GD1_WAIT_VLLM = (
    "for i in $(seq 1 40); do "
    "if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then break; fi; "
    f"if [ -f {ROOT}/jobs/vllm_openvul_grpo/exit ]; then echo 'vllm_openvul_grpo exited early'; cat {ROOT}/logs/vllm_openvul_grpo.log | tail -n 40; exit 1; fi; "
    "echo 'waiting for vllm_openvul_grpo'; sleep 30; done; "
    "curl -sf http://127.0.0.1:8000/v1/models >/dev/null"
)
preset("gd1_smoke50", [
    GD1_WAIT_VLLM,
    # 2-pair sanity gate (4 functions) testing native plain + 2-stage debate_nms before spending GPU on 50 pairs
    f"python -u llm_audit.py run --scores {GD1_SCORES} --out {ROOT}/gd1_sanity --arms plain --limit 2 --k 3 --nms_radius 3 --n_samples 1 --think_budget 3072 --answer_tokens 768 --workers 8 --tokenizer {OPENVUL_GRPO_MODEL} --plain_template_file {GD1_TPL}",
    f"python -u llm_audit.py run --scores {GD1_SCORES} --out {ROOT}/gd1_sanity --arms debate_nms --limit 2 --k 3 --nms_radius 3 --n_samples 1 --think_budget 2048 --answer_tokens 768 --workers 8 --tokenizer {OPENVUL_GRPO_MODEL}",
    f"python -u llm_audit.py score --scores {GD1_SCORES} --out {ROOT}/gd1_sanity",
    GD1_SANITY_CHECK,
    # Batch 1: plain, plain_hint_nms, plain_rand with n_samples=3
    f"T0=$(date +%s); python -u llm_audit.py run --scores {GD1_SCORES} --out {GD1_OUT} --subset_json {GD1_SUB} --arms plain plain_hint_nms plain_rand --k 3 --nms_radius 3 --n_samples 3 --think_budget 3072 --answer_tokens 768 --workers 16 --tokenizer {OPENVUL_GRPO_MODEL} --plain_template_file {GD1_TPL}; B1_ELAPSED=$(( $(date +%s) - T0 )); echo \"BATCH1_SECS=$B1_ELAPSED\"; N_DEB=1; if [ $B1_ELAPSED -lt 900 ]; then N_DEB=3; fi; echo \"USING_N_DEB=$N_DEB\"; python -u llm_audit.py run --scores {GD1_SCORES} --out {GD1_OUT} --subset_json {GD1_SUB} --arms debate_nms debate_rand debate_none --k 3 --nms_radius 3 --n_samples $N_DEB --think_budget 2048 --answer_tokens 768 --workers 16 --tokenizer {OPENVUL_GRPO_MODEL}",
    f"python -u llm_audit.py score --scores {GD1_SCORES} --out {GD1_OUT}",
    f"python -u pair_protocol.py --primevul {PV} --clean_json {GD1_LEAK} --runs {GD1_OUT}/scores/plain {GD1_OUT}/scores/plain_hint_nms {GD1_OUT}/scores/plain_rand {GD1_OUT}/scores/debate_nms {GD1_OUT}/scores/debate_rand {GD1_OUT}/scores/debate_none --out {GD1_OUT}/pair_protocol.json",
    f"python -u compare_arms.py --primevul {PV} --line none --llm {GD1_OUT} --clean_json {GD1_LEAK} --out {GD1_OUT}/compare_arms.json",
], note="(Deprecated Plan v2) L4 50-pair CLEAN smoke incl. same-model debate_* on OpenVul-GRPO", pull=["gd1_smoke50", "gd1_sanity"])

# Plan v2.1 (Phase 1A): OpenVul-Qwen3-4B-GRPO native format on A100 (--max-num-seqs 48, --workers 32)
GD1A_OUT = f"{ROOT}/gd1_smoke50a"
GD1A_LINE_DIR = f"{ROOT}/upload"
GD1A_FLAGS = (
    f"--scores {GD1_SCORES} --out {GD1A_OUT} --subset_json {GD1_SUB} "
    f"--k 3 --nms_radius 3 --n_samples 3 --think_budget 3072 --answer_tokens 768 "
    f"--workers 24 --tokenizer {OPENVUL_GRPO_MODEL} --plain_template_file {GD1_TPL}"
)
GD1A_GATE_CHECK = (
    "python - << 'PY'\n"
    "import json, sys\n"
    f"s = json.load(open('{GD1A_OUT}/scores/summary.json'))\n"
    "v = s['plain']\n"
    "m = v.get('modes', {})\n"
    "tot_m = max(sum(m.values()), 1)\n"
    "tf = 100.0 * m.get('think_forced', 0) / tot_m\n"
    "print(f'GD1A_PLAIN_GATE: answer_rate={v[\"answer_rate\"]}% think_forced={tf:.1f}% modes={m} gen_mean={v[\"gen_tokens\"][\"mean\"]}')\n"
    "if v['answer_rate'] < 90.0 or tf > 25.0:\n"
    "    sys.exit(f'GD1A_PLAIN_GATE_FAIL: answer_rate={v[\"answer_rate\"]} think_forced={tf:.1f}%')\n"
    "print('GD1A_PLAIN_GATE_PASS')\n"
    "PY"
)
GD1A_ABLATION = (
    "python - << 'PY'\n"
    "import json\n"
    "import numpy as np\n"
    "\n"
    f"out_dir = '{GD1A_OUT}'\n"
    f"ts_rows = {{r['idx']: r for r in (json.loads(l) for l in open('{GD1_SCORES}', encoding='utf-8'))}}\n"
    f"sub = json.load(open('{GD1_SUB}', encoding='utf-8'))\n"
    "pairs = []\n"
    "for a, b in sub['pair_indices']:\n"
    "    ra, rb = ts_rows[a], ts_rows[b]\n"
    "    v_idx, f_idx = (a, b) if ra['target'] == 1 else (b, a)\n"
    "    fl = list(ts_rows[v_idx].get('flaw_lines') or [])\n"
    "    pairs.append((v_idx, f_idx, fl))\n"
    "\n"
    "nms_recs = {r['idx']: r for r in (json.loads(l) for l in open(f'{out_dir}/plain_hint_nms.jsonl', encoding='utf-8'))}\n"
    "hit_flags = []\n"
    "for v_idx, _, fl in pairs:\n"
    "    if not fl:\n"
    "        hit_flags.append(None)\n"
    "    else:\n"
    "        hints = nms_recs[v_idx].get('hint_lines') or []\n"
    "        hit = any(abs((h - 1) - x) <= 2 for h in hints for x in fl)\n"
    "        hit_flags.append(hit)\n"
    "\n"
    "masks = {\n"
    "    'all': np.ones(len(pairs), dtype=bool),\n"
    "    'changed': np.array([bool(fl) for _, _, fl in pairs]),\n"
    "    'add_only': np.array([not bool(fl) for _, _, fl in pairs]),\n"
    "    'hit_hunk2': np.array([h is True for h in hit_flags]),\n"
    "    'miss_changed': np.array([h is False for h in hit_flags]),\n"
    "}\n"
    "\n"
    "def calc_metrics(sv, sf, m):\n"
    "    sv, sf = sv[m], sf[m]\n"
    "    n = int(m.sum())\n"
    "    if n == 0:\n"
    "        return {'n': 0}\n"
    "    fv, ff = sv > 0, sf > 0\n"
    "    pc = float((fv & ~ff).mean() * 100)\n"
    "    pv = float((fv & ff).mean() * 100)\n"
    "    pb = float((~fv & ~ff).mean() * 100)\n"
    "    pr = float((~fv & ff).mean() * 100)\n"
    "    rank = np.where(sv > sf, 1.0, np.where(sv == sf, 0.5, 0.0))\n"
    "    return {\n"
    "        'n': n,\n"
    "        'P-C': round(pc, 2), 'P-V': round(pv, 2), 'P-B': round(pb, 2), 'P-R': round(pr, 2),\n"
    "        'net': round(pc - pr, 2),\n"
    "        'rank_acc': round(float(rank.mean() * 100), 2),\n"
    "        'ties_pct': round(float((sv == sf).mean() * 100), 2),\n"
    "        'flag_rate': round(float(np.concatenate([fv, ff]).mean() * 100), 2),\n"
    "    }\n"
    "\n"
    "arms = ('plain', 'plain_hint_nms', 'plain_rand')\n"
    "res = {'counts': {k: int(v.sum()) for k, v in masks.items()}, 'n1_vs_n3': {}, 'stratified_n3': {}}\n"
    "for arm in arms:\n"
    "    raw = {r['idx']: r for r in (json.loads(l) for l in open(f'{out_dir}/{arm}.jsonl', encoding='utf-8'))}\n"
    "    n3_sc = {r['idx']: r['logit'] for r in (json.loads(l) for l in open(f'{out_dir}/scores/{arm}/primevul_test_scores.jsonl', encoding='utf-8'))}\n"
    "    def s1_score(idx):\n"
    "        rec = raw[idx]\n"
    "        s0 = rec['samples'][0] if isinstance(rec.get('samples'), list) and rec['samples'] else rec\n"
    "        if s0.get('status') == 'ok' and s0.get('parsed'):\n"
    "            return 1.5 if s0['parsed'].get('vulnerable') else -1.5\n"
    "        return 0.0\n"
    "    sv1 = np.array([s1_score(v) for v, _, _ in pairs])\n"
    "    sf1 = np.array([s1_score(f) for _, f, _ in pairs])\n"
    "    sv3 = np.array([n3_sc[v] for v, _, _ in pairs])\n"
    "    sf3 = np.array([n3_sc[f] for _, f, _ in pairs])\n"
    "    res['n1_vs_n3'][arm] = {}\n"
    "    for sname in ('all', 'changed', 'add_only'):\n"
    "        res['n1_vs_n3'][arm][sname] = {\n"
    "            'n1': calc_metrics(sv1, sf1, masks[sname]),\n"
    "            'n3': calc_metrics(sv3, sf3, masks[sname]),\n"
    "        }\n"
    "    res['stratified_n3'][arm] = {sname: calc_metrics(sv3, sf3, masks[sname]) for sname in ('hit_hunk2', 'miss_changed', 'add_only')}\n"
    "\n"
    "with open(f'{out_dir}/ablation_n1_vs_n3.json', 'w', encoding='utf-8') as fh:\n"
    "    json.dump(res, fh, indent=1)\n"
    "print('=== ABLATION N1 vs N3 ===')\n"
    "print(json.dumps(res, indent=1))\n"
    "PY"
)
preset("gd1_smoke50a", [
    GD1_WAIT_VLLM,
    f"python -u llm_audit.py run {GD1A_FLAGS} --arms plain",
    f"python -u llm_audit.py score --scores {GD1_SCORES} --out {GD1A_OUT}",
    GD1A_GATE_CHECK,
    f"python -u llm_audit.py run {GD1A_FLAGS} --arms plain_hint_nms plain_rand",
    f"python -u llm_audit.py score --scores {GD1_SCORES} --out {GD1A_OUT}",
    f"python -u pair_protocol.py --primevul {PV} --clean_json {GD1_LEAK} --runs {GD1A_OUT}/scores/plain {GD1A_OUT}/scores/plain_hint_nms {GD1A_OUT}/scores/plain_rand --out {GD1A_OUT}/pair_protocol.json",
    f"python -u compare_arms.py --primevul {PV} --line {GD1A_LINE_DIR} --llm {GD1A_OUT} --clean_json {GD1_LEAK} --out {GD1A_OUT}/compare_arms.json",
    GD1A_ABLATION,
], note="Plan v2.1 Phase 1A: 50-pair CLEAN smoke (plain/plain_hint_nms/plain_rand, n=3) on A100", pull=["gd1_smoke50a"])

# Plan v3 (Phase 3.0 CPU Audit + Phase 3.1 Tier 1 Frozen Qwen2.5-Coder-7B Probing)
GD3_0_OUT = f"{ROOT}/gd3_0_audit"
preset("gd3_0_audit", [
    f"python -u audit_gd3_cpu.py --primevul {PV} --out {GD3_0_OUT} --manifest {ROOT}/upload/fulltest_manifest.json --test_scores {ROOT}/upload/test_scores.jsonl --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct",
], note="Plan v3 Phase 3.0: PrimeVul train/valid/test pair, length, token & project overlap audit (CPU)", pull=["gd3_0_audit"])

GD3_1_OUT = f"{ROOT}/gd3_1_probe_qwen25_7b"
preset("gd3_1_probe_qwen25_7b", [
    f"python -u probe_llm_t1.py --primevul {PV} --out {GD3_1_OUT} --model Qwen/Qwen2.5-Coder-7B-Instruct --manifest {ROOT}/upload/fulltest_manifest.json --test_scores {ROOT}/upload/test_scores.jsonl --manual40 {ROOT}/upload/manual_labels_test40.json --max_len 2048 --extra_benign 2000 --seed 42 --cache_npz {ROOT}/tmp_emb/qwen25_7b_feats.npz",
    f"python -u pair_protocol.py --primevul {PV} --runs {GD3_1_OUT}/scores/zero_shot_yesno {GD3_1_OUT}/scores/probe_std {GD3_1_OUT}/scores/probe_len_balanced {GD3_1_OUT}/scores/probe_len_orth {GD3_1_OUT}/scores/probe_mixed_benign {GD3_1_OUT}/scores/mlp_len_orth --out {GD3_1_OUT}/pair_protocol_t1.json",
], note="Plan v3 Tier 1: Frozen Qwen2.5-Coder-7B-Instruct probing & 5-gate anti-shortcut audit (L4)", pull=["gd3_1_probe_qwen25_7b"])

GD3_2_PIP = 'python -c "import peft, bitsandbytes, accelerate" 2>/dev/null || python -m pip install -q --progress-bar off peft bitsandbytes accelerate'
GD3_2_SANITY_OUT = f"{ROOT}/gd3_2_lora_sanity"
preset("gd3_2_lora_sanity", [
    GD3_2_PIP,
    f"python -u train_lora_t2.py --primevul {PV} --out {GD3_2_SANITY_OUT} --ckpt_dir {ROOT}/tmp_ckpt/gd3_2_sanity --train_limit 200 --extra_benign 200 --valid_limit 100 --valid_benign 200 --epochs 1 --eval_every 1.0 --ckpt_every 100 --head_warmup_pairs 64 --lr 2e-5 --head_lr 5e-4 --lora_alpha 16 --matched_frac 0.85 --skip_test --no_resume --max_len 2048",
], note="Plan v3 Tier 2 sanity: 200-pair batched LP-FT Qwen2.5-7B NF4 QLoRA check (~5 min L4)", pull=["gd3_2_lora_sanity"])

GD3_2_OUT = f"{ROOT}/gd3_2_lora_nohint"
preset("gd3_2_lora_nohint", [
    GD3_2_PIP,
    f"python -u train_lora_t2.py --primevul {PV} --out {GD3_2_OUT} --ckpt_dir {ROOT}/tmp_ckpt/gd3_2_nohint --manifest {ROOT}/upload/fulltest_manifest.json --test_scores {ROOT}/upload/test_scores.jsonl --manual40 {ROOT}/upload/manual_labels_test40.json --taxonomy {ROOT}/upload/taxonomy_test_pairs.jsonl --epochs 2 --eval_every 0.5 --ckpt_every 400 --head_warmup_pairs 400 --lr 2e-5 --head_lr 5e-4 --lora_alpha 16 --max_len 2048 --extra_benign 4347 --matched_frac 0.85 --valid_benign 1000",
    f"python -u pair_protocol.py --primevul {PV} --runs {GD3_2_OUT}/scores/lora_nohint --out {GD3_2_OUT}/pair_protocol_t2.json",
], note="Plan v3 Tier 2: Paired Anti-Shortcut QLoRA (no hint, L13+L27 attn head, L4)", pull=["gd3_2_lora_nohint"])


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
