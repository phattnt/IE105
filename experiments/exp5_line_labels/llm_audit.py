"""Tier 2b of "localise, then audit the guards": an LLM judges a single function, with or without the line model's
suspicious lines as hints. Runs against an OpenAI-compatible vLLM server. Resumable: one JSONL per arm, one row per
function that got an answer; transport failures go to <arm>.errors.jsonl and are retried on the next run.

Arms (all see ONE function only, never the pair, commit or CVE):
  plain         "is this function vulnerable?"                              thinking (budgeted)
  audit         guard audit: pick sinks, list guards, missing/sufficient    thinking (budgeted)
  audit_hint    guard audit, the line model's top-k code lines as hints     thinking (budgeted)
  audit_rand    guard audit, k random code lines as hints (control)         thinking (budgeted)
  audit_hint_nt audit_hint with thinking off                                 no thinking
  audit_ctx     guard audit + related code of the repository at the parent  thinking (budgeted)
                of the fixing commit (build_context.py: callee definitions, call sites); both versions of a pair get the
                same text. Guards may come from that code (a caller's check, a callee's contract).
  audit_ctxfake audit_ctx with the control context (same files and size, functions not called + other call sites);
                the prompt is otherwise identical, so real - fake = what the content of the context adds

Thinking budget: the first call may generate --think_budget + --answer_tokens tokens (reasoning + answer). When it does
not end with a parseable answer, ALL of the reasoning it produced is kept (25/10 fix: the 24/10 version cut it back to
--think_budget, discarding up to --answer_tokens of reasoning, even reasoning that had finished), closed with </think>,
and the model answers from there (via /v1/completions on the same chat template). mode: think_complete (answered in the
first call), think_reanswer (reasoning finished, answer cut or unparsable), think_forced (reasoning hit the cap),
no_think. Every row stores the full reasoning and the full answer text, and the tokens of every call.
Sampling follows the Qwen3.5 model card: thinking temperature 0.6 / top_p 0.95 / top_k 20 (coding preset),
non-thinking 0.7 / 0.8 / 20 with presence_penalty 1.5. The per-function seed does NOT make reruns identical (vLLM
batching changes the numerics), so reruns differ in text and occasionally in verdict.
Answers that are not valid JSON are read leniently ("vulnerable": true/false and "probability": N by regex); such
rows are marked parse "lenient".
Input rows need idx, target, func; pair_idx, flaw_lines and line_scores are optional (hint arms need line_scores).

  run:   python llm_audit.py run --scores test_scores.jsonl --out <dir> --arms audit_hint ... [--limit 20] [--k 5]
         [--url http://127.0.0.1:8000/v1] [--workers 16] [--think_budget 1536] [--answer_tokens 768]
         [--tokenizer Qwen/Qwen3.5-9B | none]
  score: python llm_audit.py score --scores test_scores.jsonl --out <dir>
         -> <dir>/scores/<arm>/primevul_test_scores.jsonl for pair_protocol.py; <dir>/scores/summary.json
         score = (+1 if vulnerable else -1) + (probability - 50) / 100: the sign is the verdict (threshold 0 = the
         model's own decision), the probability only orders functions within the same verdict.
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

COMMENT = ("//", "/*", "*", "*/")
MAX_CHARS = 24000          # ~7k tokens; longer functions are cut (counted in the summary)
ARMS = (
    "plain",
    "plain_hint",
    "plain_hint_nms",
    "plain_rand",
    "audit",
    "audit_hint",
    "audit_rand",
    "audit_hint_nt",
    "audit_ctx",
    "audit_ctxfake",
    "debate_nms",
    "debate_rand",
    "debate_none",
)
DEBATE_ARMS = frozenset({"debate_nms", "debate_rand", "debate_none"})
CTX_ARMS = {"audit_ctx": "context", "audit_ctxfake": "context_fake"}
MAX_CTX_CHARS = 9000       # build_context.py --budget 8000 plus headers
SAMPLING = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}}
FORCE_CLOSE = "\n\nI have reached my thinking budget, so I will give the final answer now.\n</think>\n\n"

SYSTEM = ("You are a careful C/C++ security auditor. You see exactly one function, nothing else: no caller, no patch, "
          "no commit message. Decide from the code alone. Be precise about line numbers. Answer with a single JSON object "
          "and nothing after it.")

PLAIN = """Does this function contain a security vulnerability (memory safety, integer overflow, missing validation, use-after-free, race, etc.)?

```c
{code}
```

Answer with JSON only: {{"vulnerable": true or false, "probability": <integer 0-100, your probability that the function is vulnerable>, "reason": "one sentence"}}"""

PLAIN_HINT = """Does this function contain a security vulnerability (memory safety, integer overflow, missing validation, use-after-free, race, etc.)?
Lines are numbered.

```c
{code}
```
{hint}
Answer with JSON only: {{"vulnerable": true or false, "probability": <integer 0-100, your probability that the function is vulnerable>, "reason": "one sentence"}}"""

AUDIT = """Audit this function for a security vulnerability by checking whether its dangerous operations are guarded.
Lines are numbered.

```c
{code}
```
{hint}
Procedure:
1. Choose up to {k} candidate lines that perform a dangerous operation: memory read/write through an index or pointer, length/size arithmetic, allocation size, copy/format calls, pointer dereference after a lookup, integer conversion, lock/refcount handling.{hint_rule}
2. For each candidate, name the operands that must be validated for the operation to be safe, then list the line numbers of the checks IN THIS FUNCTION that validate those operands on the control path before the operation (if/while/for conditions, early returns, asserts, min/max clamps). Judge "missing" when no listed check makes the operation safe (absent check, wrong bound, off-by-one, wrong direction, checked variable differs from the used one), otherwise "sufficient".
3. The function is vulnerable if and only if at least one candidate is "missing".

Answer with JSON only:
{{"sinks": [{{"line": <int>, "operation": "<short>", "operands": ["<name>", ...], "guards": [<int>, ...], "verdict": "missing" or "sufficient", "reason": "<one sentence>"}}, ...],
 "vulnerable": true or false, "probability": <integer 0-100, your probability that the function is vulnerable>}}"""

HINT = "\nA line-level model trained on vulnerability fixes flagged these lines as the most suspicious: {lines}.\n"
HINT_RULE = " Audit the flagged lines first; you may add or drop candidates."

HINT_NEUTRAL = "\nA static analysis pass identified candidate lines performing sensitive/critical operations: {lines}.\n"
HINT_RULE_NEUTRAL = " Check whether operations on these lines are properly guarded; you may add or drop candidates."

HINT_PLAIN_NEUTRAL = (
    "\nNote: A lightweight static pass flagged these candidate lines for closer inspection: {lines}. "
    "They may already be properly guarded or completely harmless; judge the entire function independently.\n"
)

PROSECUTOR_PROMPT = """Analyze this C/C++ function for a real, self-contained security vulnerability (memory safety, bounds error, integer overflow/underflow, null/dangling dereference, uninitialized use, double-free/use-after-free, command/format injection, race condition).
Lines are numbered.

```c
{code}
```
{hint}
Rules:
1. Check whether any operation in this function can trigger a security bug through flawed logic or missing validation in THIS function itself.
2. Do NOT flag a function solely because caller arguments are unchecked unless the function's own logic shows a concrete exploitable defect.
3. If vulnerable, state the single most concrete claim (`line`, `cwe`, `variable`, `trigger` condition). If safe or properly guarded, set `"vulnerable": false` and `"claim": null`.

Answer with JSON only: {{"vulnerable": true or false, "probability": <integer 0-100, probability the function is vulnerable>, "claim": {{"line": <int>, "cwe": "<short>", "variable": "<name>", "trigger": "<concrete condition causing crash/UB/exploit>"}} or null}}"""

DEFENSE_SYSTEM = (
    "You are an independent, skeptical C/C++ security reviewer verifying whether a vulnerability claim is real or a false positive. "
    "You see only this function: no caller, no patch, no commit message. Be precise about line numbers. "
    "Answer with a single JSON object and nothing after it."
)

DEFENSE_PROMPT = """Another auditor flagged a potential vulnerability in this C/C++ function. Independently verify whether the claim is actually exploitable or a false positive.
Lines are numbered.

```c
{code}
```

Reported claim:
```json
{claim}
```

Check strictly:
1. Does any check, guard, bounds clamp, early return, or safe API on the control path BEFORE the claimed line already prevent the trigger? If so, list the exact `guard_lines`.
2. Does the claim rely on an unproven assumption about external callers, invariants, or dead/unreachable paths rather than a concrete defect in this function?
Judge `"confirmed": true` ONLY if the trigger is genuinely reachable and unguarded in this function; otherwise `"confirmed": false`.

Answer with JSON only: {{"confirmed": true or false, "probability": <integer 0-100, probability the vulnerability is genuinely real and unguarded>, "guard_lines": [<int>, ...], "reason": "<one sentence>"}}"""

SYSTEM_CTX = ("You are a careful C/C++ security auditor. You audit exactly one function. You also get some related code from "
              "the same repository (definitions of functions it calls, places where it is called); it may or may not be "
              "relevant. There is no patch and no commit message. Be precise about line numbers. Answer with a single JSON "
              "object and nothing after it.")

CTX = """
Related code from the same repository (definitions of functions called here, call sites of this function). Not numbered; it is context only:

```c
{context}
```
"""

AUDIT_CTX_STEP2 = ("2. For each candidate, name the operands that must be validated for the operation to be safe, then list the checks "
                   "that validate those operands before the operation: line numbers in this function (if/while/for conditions, "
                   "early returns, asserts, min/max clamps), or a short quote of a guarantee in the related code (a check in "
                   "a caller, a callee that cannot return NULL / bounds its result). Judge \"missing\" when no listed check makes "
                   "the operation safe (absent check, wrong bound, off-by-one, wrong direction, checked variable differs from "
                   "the used one), otherwise \"sufficient\".")


def is_code(line):
    s = line.strip()
    return bool(s) and not s.startswith(COMMENT)


def numbered(func):
    lines = func.split("\n")
    cut = False
    if len(func) > MAX_CHARS:
        keep, total = [], 0
        for l in lines:
            if total + len(l) + 1 > MAX_CHARS:
                cut = True
                break
            keep.append(l); total += len(l) + 1
        lines = keep
    w = len(str(len(lines)))
    return "\n".join(f"{i + 1:{w}d}| {l}" for i, l in enumerate(lines)), len(lines), cut


def hint_lines(r, arm, k, n_kept, nms_radius=0):
    lines = r["func"].split("\n")
    code = [i for i in range(min(len(lines), len(r["line_scores"]), n_kept)) if is_code(lines[i])]
    if arm in ("audit_rand", "plain_rand", "debate_rand"):
        rng = random.Random(f"{r['idx']}")
        pick = sorted(rng.sample(code, min(k, len(code))))
    else:
        eff_r = nms_radius if nms_radius > 0 else (3 if arm.endswith("_nms") else 0)
        pick = None
        if eff_r > 0 and code:
            try:
                from clean_diff_labels import diverse_topk_order
                sub_func = "\n".join(lines[:n_kept])
                ordered = diverse_topk_order(sub_func, list(r["line_scores"][:n_kept]), radius=eff_r, mask_trivial=True)
                code_set = set(code)
                filtered = [i for i in ordered if i in code_set]
                if len(filtered) >= min(k, len(code)):
                    pick = sorted(filtered[:k])
            except Exception:
                pick = None
        if pick is None:
            pick = sorted(sorted(code, key=lambda i: (-r["line_scores"][i], i))[:k])
    return [i + 1 for i in pick]          # 1-based in the prompt


def build_defense(r, claim):
    """Build stage-2 independent defense/falsifier messages (never sees stage-1 reasoning)."""
    code, _, _ = numbered(r["func"])
    claim_str = json.dumps(claim, ensure_ascii=False, indent=1)
    user = DEFENSE_PROMPT.format(code=code, claim=claim_str)
    return [{"role": "system", "content": DEFENSE_SYSTEM}, {"role": "user", "content": user}]


def build(r, arm, k, hint_style="neutral", nms_radius=0, plain_templates=None, system_override=None):
    """-> (messages, meta, thinking)"""
    code, n_kept, cut = numbered(r["func"])
    meta = {"idx": r["idx"], "arm": arm, "cut": cut, "n_lines": n_kept}
    tpls = plain_templates or {}
    sys_msg = system_override or tpls.get("system") or SYSTEM
    if arm == "plain":
        user = (tpls.get("plain") or PLAIN).format(code=code)
    elif arm in ("plain_hint", "plain_hint_nms", "plain_rand"):
        meta["hint_lines"] = hint_lines(r, arm, k, n_kept, nms_radius=nms_radius)
        if arm in ("plain_hint_nms", "plain_rand"):
            h_tmpl = tpls.get("hint_plain_neutral") or HINT_PLAIN_NEUTRAL
        else:
            h_tmpl = HINT_NEUTRAL if hint_style == "neutral" else HINT
        hint = h_tmpl.format(lines=", ".join(map(str, meta["hint_lines"])))
        user = (tpls.get("plain_hint") or PLAIN_HINT).format(code=code, hint=hint)
    elif arm in DEBATE_ARMS:
        hint = ""
        if arm in ("debate_nms", "debate_rand"):
            meta["hint_lines"] = hint_lines(r, arm, k, n_kept, nms_radius=nms_radius)
            h_tmpl = tpls.get("hint_plain_neutral") or HINT_PLAIN_NEUTRAL
            hint = h_tmpl.format(lines=", ".join(map(str, meta["hint_lines"])))
        user = PROSECUTOR_PROMPT.format(code=code, hint=hint)
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], meta, True
    else:
        hint, rule = "", ""
        if arm in ("audit_hint", "audit_rand", "audit_hint_nt"):
            meta["hint_lines"] = hint_lines(r, arm, k, n_kept, nms_radius=nms_radius)
            h_tmpl = HINT_NEUTRAL if hint_style == "neutral" else HINT
            r_tmpl = HINT_RULE_NEUTRAL if hint_style == "neutral" else HINT_RULE
            hint, rule = h_tmpl.format(lines=", ".join(map(str, meta["hint_lines"]))), r_tmpl
        if arm in CTX_ARMS:
            ctx = (r.get(CTX_ARMS[arm]) or "").strip()[:MAX_CTX_CHARS] or "/* nothing found */"
            meta["ctx_chars"] = len(ctx)
            hint = CTX.format(context=ctx)
        user = AUDIT.format(code=code, k=k, hint=hint, hint_rule=rule)
        if arm in CTX_ARMS:
            step2 = re.search(r"^2\. .*$", user, re.M)
            user = user[:step2.start()] + AUDIT_CTX_STEP2 + user[step2.end():]
            user = user.replace('"guards": [<int>, ...]', '"guards": [<int or "quote">, ...]')
            return [{"role": "system", "content": SYSTEM_CTX}, {"role": "user", "content": user}], meta, True
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], meta, arm != "audit_hint_nt"


def parse_json(text):
    """last parseable {...} object in the text (tolerant to fences, prose, trailing commas); None if there is none."""
    text = (text or "").strip()
    for c in reversed(re.findall(r"\{.*\}", text, flags=re.S)):
        for s in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                return json.loads(s)
            except Exception:
                continue
    start = text.rfind("{")
    while start >= 0:
        try:
            return json.loads(text[start:])
        except Exception:
            start = text.rfind("{", 0, start)
    return None


def parse_answer(text, allow_tag=True):
    """-> (dict with "vulnerable", how) where how is "json", "lenient", or "tag"; (None, None) when no verdict can be read."""
    js = parse_json(text)
    if isinstance(js, dict) and "vulnerable" in js:
        return js, "json"
    t = text or ""
    v = re.findall(r'"?vulnerable"?\s*[:=]\s*"?(true|false)', t, flags=re.I)
    if v:
        out = {"vulnerable": v[-1].lower() == "true"}
        p = re.findall(r'"?probability"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', t, flags=re.I)
        if p:
            out["probability"] = float(p[-1])
        return out, "lenient"
    if allow_tag:
        tags = re.findall(r"\b(HAS_VUL|NO_VUL)\b", t)
        if tags:
            vul = tags[-1] == "HAS_VUL"
            return {"vulnerable": vul, "probability": 100.0 if vul else 0.0}, "tag"
    return None, None


def parse_answer_json(text):
    return parse_answer(text, allow_tag=False)


def parse_defense(text):
    """-> (dict with "confirmed", how) where how is "json" or "lenient" (or *_vul_fallback); (None, None) on failure."""
    js = parse_json(text)
    if isinstance(js, dict):
        if "confirmed" in js:
            return js, "json"
        if "vulnerable" in js:
            out = dict(js)
            out["confirmed"] = bool(js["vulnerable"])
            return out, "json_vul_fallback"
    t = text or ""
    c = re.findall(r'"?confirmed"?\s*[:=]\s*"?(true|false)', t, flags=re.I)
    how = "lenient"
    if not c:
        c = re.findall(r'"?vulnerable"?\s*[:=]\s*"?(true|false)', t, flags=re.I)
        how = "lenient_vul_fallback"
    if not c:
        return None, None
    out = {"confirmed": c[-1].lower() == "true"}
    p = re.findall(r'"?probability"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', t, flags=re.I)
    if p:
        out["probability"] = float(p[-1])
    return out, how


class Client:
    def __init__(self, a):
        import requests
        self.s = requests.Session()
        self.url, self.model, self.timeout, self.retries = a.url, a.model, a.timeout, a.retries
        self.think_budget, self.answer_tokens, self.max_model_len = a.think_budget, a.answer_tokens, a.max_model_len
        self.tok, self.tok_lock = None, threading.Lock()
        if a.tokenizer and a.tokenizer.lower() != "none":
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(a.tokenizer)

    def post(self, path, body):
        """POST with retries on transport errors and 5xx; raises after the last attempt."""
        err = None
        for i in range(self.retries):
            try:
                resp = self.s.post(f"{self.url}/{path}", json=body, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:          # noqa: BLE001 - every transport failure is retried, then re-raised
                err = e
                time.sleep(5 * (i + 1))
        raise RuntimeError(f"{path} failed after {self.retries} attempts: {err!r}"[:600])

    def chat(self, messages, thinking, max_tokens, seed):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": thinking}, **SAMPLING[thinking]}
        j = self.post("chat/completions", body)
        m = j["choices"][0]["message"]
        return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage") or {}}

    def template(self, messages):
        if self.tok is not None:
            with self.tok_lock:
                p = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        else:                               # ChatML fallback (Qwen's format), used when no tokenizer is available
            p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
        return p if p.rstrip().endswith("<think>") else p + "<think>\n"

    def cut_tokens(self, text, n):
        if self.tok is None:
            return text[:4 * n]
        with self.tok_lock:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            return text if len(ids) <= n else self.tok.decode(ids[:n])

    def forced_answer(self, messages, reasoning, finished, seed):
        """answer after the given reasoning (all of it is kept). finished: the model had closed its reasoning itself."""
        close = "\n</think>\n\n" if finished else FORCE_CLOSE
        head = self.template(messages)
        if self.tok is not None:          # 25/10: a long function + 8960 reasoning tokens overflowed --max_model_len (HTTP 400)
            with self.tok_lock:
                n_head = len(self.tok(head + close, add_special_tokens=False)["input_ids"])
                ids = self.tok(reasoning, add_special_tokens=False)["input_ids"]
                room = self.max_model_len - self.answer_tokens - n_head - 32
                if len(ids) > room:
                    reasoning = self.tok.decode(ids[:max(room, 0)])
        prompt = head + reasoning.rstrip() + close
        # the answer after a long (often looping) reasoning degenerated into repeated tokens in the 25/10 smoke; the
        # non-thinking preset's presence_penalty 1.5 (Qwen's advice against endless repetition) is used for this part
        body = {"model": self.model, "prompt": prompt, "max_tokens": self.answer_tokens, "seed": seed, **SAMPLING[False]}
        j = self.post("completions", body)
        return {"content": j["choices"][0].get("text") or "", "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage") or {}}

    def judge(self, messages, thinking, seed, parser=parse_answer):
        """-> (answer text, record of every call incl. the full reasoning). Never mixes modes silently."""
        t = time.time()
        if not thinking:
            c = self.chat(messages, False, self.answer_tokens, seed)
            return c["content"], {"mode": "no_think", "calls": [{"kind": "chat_no_think", "finish": c["finish"], "usage": c["usage"]}],
                                  "secs": round(time.time() - t, 1), "reasoning": ""}
        c = self.chat(messages, True, self.think_budget + self.answer_tokens, seed)
        calls = [{"kind": "chat_think", "finish": c["finish"], "usage": c["usage"], "reasoning_chars": len(c["reasoning"])}]
        if c["finish"] == "stop" and parser(c["content"])[0] is not None:
            return c["content"], {"mode": "think_complete", "calls": calls, "secs": round(time.time() - t, 1), "reasoning": c["reasoning"]}
        finished = bool(c["content"].strip())          # text after </think> exists: the reasoning had ended
        f = self.forced_answer(messages, c["reasoning"], finished, seed)
        calls.append({"kind": "forced_answer", "finish": f["finish"], "usage": f["usage"]})
        if parser(f["content"])[0] is None:          # one retry of the answer only, with another seed
            f = self.forced_answer(messages, c["reasoning"], finished, seed + 1)
            calls.append({"kind": "forced_answer_retry", "finish": f["finish"], "usage": f["usage"]})
        return f["content"], {"mode": "think_reanswer" if finished else "think_forced", "calls": calls,
                              "secs": round(time.time() - t, 1), "reasoning": c["reasoning"], "first_answer": c["content"]}


def seed_of(idx):
    return int(hashlib.md5(str(idx).encode()).hexdigest()[:8], 16)


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _clamp_prob(val, default=50.0):
    try:
        p = float(val)
    except Exception:
        p = float(default)
    return min(max(p, 0.0), 100.0)


def _load_subset_ids(path):
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict):
        items = obj.get("pair_indices") or obj.get("clean_pair_indices") or obj.get("indices") or []
    else:
        items = obj
    ids = set()
    for it in items:
        if isinstance(it, (list, tuple)):
            ids.update(it)
        else:
            ids.add(it)
    return ids


def _load_plain_templates(path):
    if not path:
        return None
    return json.load(open(path, encoding="utf-8"))


def _resolve_system_override(val):
    if not val:
        return None
    if os.path.isfile(val):
        return open(val, encoding="utf-8").read().strip()
    return val


def _run_one_sample(cl, r, arm, k, hint_style, nms_radius, plain_templates, system_override, sample_seed):
    messages, meta, thinking = build(
        r, arm, k, hint_style=hint_style, nms_radius=nms_radius,
        plain_templates=plain_templates, system_override=system_override,
    )
    if arm not in DEBATE_ARMS:
        text, rec = cl.judge(messages, thinking, sample_seed, parser=parse_answer)
        js, how = parse_answer(text)
        return meta, {
            **rec, "seed": sample_seed, "status": "ok" if js else "unparsed",
            "parse": how, "parsed": js, "raw": text,
        }

    text1, rec1 = cl.judge(messages, thinking, sample_seed, parser=parse_answer_json)
    pros_js, pros_how = parse_answer(text1, allow_tag=True)
    st1 = {
        "mode": rec1["mode"], "calls": rec1["calls"], "secs": rec1["secs"],
        "parse": pros_how, "parsed": pros_js, "raw": text1, "reasoning": rec1["reasoning"],
    }
    if pros_js is None:
        return meta, {
            "mode": rec1["mode"], "calls": rec1["calls"], "secs": rec1["secs"], "seed": sample_seed,
            "status": "unparsed", "parse": None, "parsed": None, "raw": text1,
            "reasoning": rec1["reasoning"], "stage1": st1, "stage2": None,
        }

    pros_vul = bool(pros_js.get("vulnerable"))
    pros_prob = _clamp_prob(pros_js.get("probability", 50.0))
    claim_obj = pros_js.get("claim")
    if not pros_vul:
        sinks = (
            [{"line": int(claim_obj["line"]), "verdict": "sufficient"}]
            if isinstance(claim_obj, dict) and isinstance(claim_obj.get("line"), (int, float))
            else []
        )
        final_js = {
            "vulnerable": False, "probability": pros_prob, "defense_skipped": True,
            "prosecutor_vulnerable": False, "claim": claim_obj, "sinks": sinks,
        }
        return meta, {
            "mode": rec1["mode"], "calls": rec1["calls"], "secs": rec1["secs"], "seed": sample_seed,
            "status": "ok", "parse": pros_how, "parsed": final_js, "raw": text1,
            "reasoning": rec1["reasoning"], "stage1": st1, "stage2": None,
        }

    claim_dict = (
        claim_obj if isinstance(claim_obj, dict)
        else {"line": None, "summary": pros_js.get("reason") or (text1 or "")[:300]}
    )
    def_msgs = build_defense(r, claim_dict)
    text2, rec2 = cl.judge(def_msgs, True, sample_seed + 17, parser=parse_defense)
    def_js, def_how = parse_defense(text2)
    st2 = {
        "mode": rec2["mode"], "calls": rec2["calls"], "secs": rec2["secs"],
        "parse": def_how, "parsed": def_js, "raw": text2, "reasoning": rec2["reasoning"],
    }
    calls = rec1["calls"] + rec2["calls"]
    secs = round(rec1["secs"] + rec2["secs"], 1)
    if "think_forced" in (rec1["mode"], rec2["mode"]):
        mode = "think_forced"
    elif "think_reanswer" in (rec1["mode"], rec2["mode"]):
        mode = "think_reanswer"
    else:
        mode = "think_complete"
    if def_js is None:
        return meta, {
            "mode": mode, "calls": calls, "secs": secs, "seed": sample_seed,
            "status": "unparsed", "parse": None, "parsed": None, "raw": text2,
            "reasoning": rec1["reasoning"], "stage1": st1, "stage2": st2,
        }

    confirmed = bool(def_js.get("confirmed"))
    def_prob = _clamp_prob(def_js.get("probability", 50.0))
    sinks = (
        [{"line": int(claim_dict["line"]), "verdict": "missing" if confirmed else "sufficient"}]
        if isinstance(claim_dict.get("line"), (int, float))
        else []
    )
    final_js = {
        "vulnerable": confirmed, "probability": def_prob, "defense_skipped": False,
        "prosecutor_vulnerable": True, "prosecutor_probability": pros_prob,
        "claim": claim_dict, "guard_lines": def_js.get("guard_lines", []),
        "reason": def_js.get("reason", ""), "sinks": sinks,
    }
    return meta, {
        "mode": mode, "calls": calls, "secs": secs, "seed": sample_seed,
        "status": "ok", "parse": f"{pros_how}+{def_how}", "parsed": final_js, "raw": text2,
        "reasoning": rec1["reasoning"], "stage1": st1, "stage2": st2,
    }


def run(a):
    rows = load_rows(a.scores)
    if getattr(a, "subset_json", None):
        ids = _load_subset_ids(a.subset_json)
        rows = [r for r in rows if r["idx"] in ids]
    if a.limit:
        vul = [r for r in rows if r["target"] == 1][:a.limit]
        ids = {r["idx"] for r in vul} | {r.get("pair_idx") for r in vul}
        rows = [r for r in rows if r["idx"] in ids]
    plain_templates = _load_plain_templates(getattr(a, "plain_template_file", None))
    system_override = _resolve_system_override(getattr(a, "system_override", None))
    n_samples = max(1, int(getattr(a, "n_samples", 1)))
    nms_radius = int(getattr(a, "nms_radius", 0))
    os.makedirs(a.out, exist_ok=True)
    cl = Client(a)
    for arm in a.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        path, epath = f"{a.out}/{arm}.jsonl", f"{a.out}/{arm}.errors.jsonl"
        done = {json.loads(l)["idx"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = [r for r in rows if r["idx"] not in done]
        print(f"[{arm}] {len(done)} done, {len(todo)} to do (n_samples={n_samples}, nms_radius={nms_radius})", flush=True)
        fh, eh = open(path, "a", encoding="utf-8"), open(epath, "a", encoding="utf-8")
        T = time.time(); n = n_err = 0; toks = 0

        def work(r):
            base_seed = seed_of(r["idx"])
            hint_style = getattr(a, "hint_style", "neutral")
            if n_samples == 1:
                meta, rec = _run_one_sample(
                    cl, r, arm, a.k, hint_style, nms_radius, plain_templates, system_override, base_seed
                )
                return {**meta, **rec, "target": r["target"], "pair_idx": r.get("pair_idx")}
            samples = []
            meta = None
            for s_idx in range(n_samples):
                m, s_rec = _run_one_sample(
                    cl, r, arm, a.k, hint_style, nms_radius, plain_templates, system_override, base_seed + 1009 * s_idx
                )
                meta = m
                samples.append(s_rec)
            valid = [x for x in samples if x["status"] == "ok" and x["parsed"]]
            all_calls = [c for x in samples for c in x["calls"]]
            total_secs = round(sum(x["secs"] for x in samples), 1)
            uniq_modes = {x["mode"] for x in samples}
            agg_mode = next(iter(uniq_modes)) if len(uniq_modes) == 1 else "mixed"
            if valid:
                vr = sum(bool(x["parsed"]["vulnerable"]) for x in valid) / len(valid)
                mean_p = sum(_clamp_prob(x["parsed"].get("probability", 50.0)) for x in valid) / len(valid)
                sinks = [s for x in valid for s in (x["parsed"].get("sinks") or [])]
                agg_parsed = {
                    "vulnerable": vr > 0.5,
                    "probability": round(mean_p, 1),
                    "vote_rate": round(vr, 4),
                    "n_valid": len(valid),
                    "n_samples": n_samples,
                }
                if sinks:
                    agg_parsed["sinks"] = sinks
                status = "ok"
                parse = "multi"
            else:
                agg_parsed, status, parse = None, "unparsed", None
            return {
                **meta, "mode": agg_mode, "calls": all_calls, "secs": total_secs,
                "target": r["target"], "pair_idx": r.get("pair_idx"),
                "status": status, "parse": parse, "parsed": agg_parsed,
                "n_samples": n_samples, "samples": samples,
            }

        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            for f in as_completed(futs):
                r = futs[f]
                try:
                    rec = f.result()
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                    toks += sum((c.get("usage") or {}).get("completion_tokens") or 0 for c in rec["calls"])
                except Exception as e:      # noqa: BLE001 - not marked done: retried on the next run
                    n_err += 1
                    eh.write(json.dumps({"idx": r["idx"], "arm": arm, "time": time.strftime("%H:%M:%S"), "error": repr(e)[:600]}) + "\n"); eh.flush()
                n += 1
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - T
                    print(f"[{arm}] {n}/{len(todo)} | errors {n_err} | {el / 60:.1f} min | {n / el * 60:.1f} fn/min | "
                          f"{toks / el:.0f} gen tok/s | ETA {(len(todo) - n) / max(n, 1) * el / 60:.0f} min", flush=True)
        fh.close(); eh.close()
        if n_err:
            print(f"[{arm}] {n_err} functions failed (see {epath}); rerun to retry them", flush=True)
    print("RUN_DONE", flush=True)


def to_score(js):
    if "vote_rate" in js:
        vr = float(js["vote_rate"])
        p = _clamp_prob(js.get("probability", 50.0))
        v = bool(js.get("vulnerable"))
        return (2.0 * vr - 1.0) + 0.1 * (p - 50.0) / 100.0, v, p
    v = bool(js.get("vulnerable"))
    p = _clamp_prob(js.get("probability", 50.0))
    return (1.0 if v else -1.0) + (p - 50.0) / 100.0, v, p


def score(a):
    rows = {r["idx"]: r for r in load_rows(a.scores)}
    summary = {}
    for path in sorted(os.listdir(a.out)):
        if not path.endswith(".jsonl") or path.endswith(".errors.jsonl"):
            continue
        arm = path[:-6]
        recs = {}
        for x in load_rows(f"{a.out}/{path}"):            # last row per function wins
            recs[x["idx"]] = x
        recs = list(recs.values())
        d = f"{a.out}/scores/{arm}"; os.makedirs(d, exist_ok=True)
        modes, n_ok, n_vul, n_cut, consistent, n_len = {}, 0, 0, 0, 0, 0
        n_fl = hit_missing = hit_sink = hit_hint = n_miss_total = 0
        gen = []
        s1_modes, s2_modes = {}, {}
        n_rollouts = n_pros_vul = n_def_called = n_def_rej = 0
        with open(f"{d}/primevul_test_scores.jsonl", "w", encoding="utf-8") as fh:
            for x in recs:
                rollouts = x.get("samples") if isinstance(x.get("samples"), list) else [x]
                for sr in rollouts:
                    m = sr.get("mode", "?")
                    modes[m] = modes.get(m, 0) + 1
                    if "stage1" in sr:
                        n_rollouts += 1
                        st1 = sr.get("stage1") or {}
                        st2 = sr.get("stage2")
                        m1 = st1.get("mode", "?")
                        s1_modes[m1] = s1_modes.get(m1, 0) + 1
                        if st1.get("parsed") and bool(st1["parsed"].get("vulnerable")):
                            n_pros_vul += 1
                        if st2:
                            n_def_called += 1
                            m2 = st2.get("mode", "?")
                            s2_modes[m2] = s2_modes.get(m2, 0) + 1
                            if st2.get("parsed") and not bool(st2["parsed"].get("confirmed")):
                                n_def_rej += 1
                n_cut += bool(x.get("cut"))
                gen.append(sum((c.get("usage") or {}).get("completion_tokens") or 0 for c in x.get("calls", [])))
                js = x.get("parsed")
                if x.get("status") == "ok" and js:
                    s, v, p = to_score(js)
                    n_ok += 1; n_vul += v; consistent += (v == (p > 50)) or p == 50; n_len += x.get("parse") == "lenient"
                else:
                    s = 0.0                                   # no answer: not flagged (counts as a miss), reported
                fh.write(json.dumps({"idx": x["idx"], "target": x["target"], "logit": s}) + "\n")
                r = rows.get(x["idx"])
                if r and r["target"] == 1 and r.get("flaw_lines") and js and isinstance(js.get("sinks"), list):
                    n_fl += 1
                    fl = set(r["flaw_lines"])
                    sinks = [t for t in js["sinks"] if isinstance(t, dict) and isinstance(t.get("line"), (int, float))]
                    miss = [int(t["line"]) - 1 for t in sinks if str(t.get("verdict", "")).lower().startswith("miss")]
                    n_miss_total += len(miss)
                    hit_missing += any(i in fl for i in miss)
                    hit_sink += any(int(t["line"]) - 1 in fl for t in sinks)
                    if x.get("hint_lines"):
                        hit_hint += any(i + 1 in set(x["hint_lines"]) for i in miss)
        n = len(recs)
        answered_ids = {x["idx"] for x in recs}
        err_ids = {json.loads(l)["idx"] for l in open(f"{a.out}/{arm}.errors.jsonl")} if os.path.exists(f"{a.out}/{arm}.errors.jsonl") else set()
        errors = {"logged": len(err_ids), "still_unanswered": len(err_ids - answered_ids)}
        gen.sort()
        n_samp_max = max((x.get("n_samples", 1) for x in recs), default=1)
        summary[arm] = {"n": n, "n_samples": n_samp_max, "answered": n_ok, "answer_rate": round(n_ok / max(n, 1) * 100, 1),
                        "parsed_leniently": n_len, "modes": modes,
                        "transport_errors": errors, "cut_functions": n_cut,
                        "flag_rate": round(n_vul / max(n_ok, 1) * 100, 1),
                        "verdict_probability_consistent": round(consistent / max(n_ok, 1) * 100, 1),
                        "gen_tokens": {"mean": round(sum(gen) / max(n, 1)), "median": gen[n // 2] if gen else 0, "max": gen[-1] if gen else 0},
                        "lines": {"n_vul_with_flaw_lines": n_fl,
                                  "a_missing_sink_is_a_flaw_line": round(hit_missing / max(n_fl, 1) * 100, 1),
                                  "any_sink_is_a_flaw_line": round(hit_sink / max(n_fl, 1) * 100, 1),
                                  "a_missing_sink_is_a_hint_line": round(hit_hint / max(n_fl, 1) * 100, 1),
                                  "missing_sinks_per_function": round(n_miss_total / max(n_fl, 1), 2)}}
        if n_rollouts:
            summary[arm]["debate_stats"] = {
                "n_rollouts": n_rollouts,
                "prosecutor_flag_rate": round(n_pros_vul / max(n_rollouts, 1) * 100, 1),
                "defense_called": n_def_called,
                "defense_rejected_rate": round(n_def_rej / max(n_def_called, 1) * 100, 1),
                "stage1_modes": s1_modes,
                "stage2_modes": s2_modes,
            }
        print(arm, json.dumps(summary[arm]))
    json.dump(summary, open(f"{a.out}/scores/summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=["audit_hint"])
    ap.add_argument("--limit", type=int, default=None, help="smoke: this many vulnerable functions (with flaw lines) + their pairs")
    ap.add_argument("--subset_json", default=None, help="JSON file with pair_indices / clean_pair_indices / idx list to filter rows")
    ap.add_argument("--k", type=int, default=3, help="top-k hint lines (default: 3 to avoid OR-gate false alarm explosion)")
    ap.add_argument("--nms_radius", type=int, default=0, help="1D NMS radius for hint selection (0=raw top-k; *_nms arms default to 3)")
    ap.add_argument("--n_samples", type=int, default=1, help="number of rollouts per function for self-consistency")
    ap.add_argument("--hint_style", choices=["neutral", "asymmetric"], default="neutral", help="symmetric neutral hints vs asymmetric priming")
    ap.add_argument("--system_override", default=None, help="override SYSTEM prompt (string or path to text file)")
    ap.add_argument("--plain_template_file", default=None, help="JSON file overriding plain/plain_hint/hint_plain_neutral/system templates")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="llm")
    ap.add_argument("--workers", type=int, default=16, help="keep <= the server's --max-num-seqs so requests do not queue")
    ap.add_argument("--think_budget", type=int, default=1536)
    ap.add_argument("--answer_tokens", type=int, default=768)
    ap.add_argument("--max_model_len", type=int, default=20480, help="the server's --max-model-len (forced answers are fitted into it)")
    ap.add_argument("--timeout", type=int, default=1800, help="per HTTP request, seconds")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B", help="for budget forcing; 'none' = ChatML + 4 chars/token")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
