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
ARMS = ("plain", "audit", "audit_hint", "audit_rand", "audit_hint_nt", "audit_ctx", "audit_ctxfake")
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


def hint_lines(r, arm, k, n_kept):
    lines = r["func"].split("\n")
    code = [i for i in range(min(len(lines), len(r["line_scores"]), n_kept)) if is_code(lines[i])]
    if arm == "audit_rand":
        rng = random.Random(f"{r['idx']}")
        pick = sorted(rng.sample(code, min(k, len(code))))
    else:
        pick = sorted(sorted(code, key=lambda i: -r["line_scores"][i])[:k])
    return [i + 1 for i in pick]          # 1-based in the prompt


def build(r, arm, k, hint_style="neutral"):
    """-> (messages, meta, thinking)"""
    code, n_kept, cut = numbered(r["func"])
    meta = {"idx": r["idx"], "arm": arm, "cut": cut, "n_lines": n_kept}
    if arm == "plain":
        user = PLAIN.format(code=code)
    else:
        hint, rule = "", ""
        if arm in ("audit_hint", "audit_rand", "audit_hint_nt"):
            meta["hint_lines"] = hint_lines(r, arm, k, n_kept)
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
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], meta, arm != "audit_hint_nt"


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


def parse_answer(text):
    """-> (dict with "vulnerable", how) where how is "json" or "lenient"; (None, None) when no verdict can be read."""
    js = parse_json(text)
    if isinstance(js, dict) and "vulnerable" in js:
        return js, "json"
    t = text or ""
    v = re.findall(r'"?vulnerable"?\s*[:=]\s*"?(true|false)', t, flags=re.I)
    if not v:
        return None, None
    out = {"vulnerable": v[-1].lower() == "true"}
    p = re.findall(r'"?probability"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', t, flags=re.I)
    if p:
        out["probability"] = float(p[-1])
    return out, "lenient"


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

    def judge(self, messages, thinking, seed):
        """-> (answer text, record of every call incl. the full reasoning). Never mixes modes silently."""
        t = time.time()
        if not thinking:
            c = self.chat(messages, False, self.answer_tokens, seed)
            return c["content"], {"mode": "no_think", "calls": [{"kind": "chat_no_think", "finish": c["finish"], "usage": c["usage"]}],
                                  "secs": round(time.time() - t, 1), "reasoning": ""}
        c = self.chat(messages, True, self.think_budget + self.answer_tokens, seed)
        calls = [{"kind": "chat_think", "finish": c["finish"], "usage": c["usage"], "reasoning_chars": len(c["reasoning"])}]
        if c["finish"] == "stop" and parse_answer(c["content"])[0] is not None:
            return c["content"], {"mode": "think_complete", "calls": calls, "secs": round(time.time() - t, 1), "reasoning": c["reasoning"]}
        finished = bool(c["content"].strip())          # text after </think> exists: the reasoning had ended
        f = self.forced_answer(messages, c["reasoning"], finished, seed)
        calls.append({"kind": "forced_answer", "finish": f["finish"], "usage": f["usage"]})
        if parse_answer(f["content"])[0] is None:          # one retry of the answer only, with another seed
            f = self.forced_answer(messages, c["reasoning"], finished, seed + 1)
            calls.append({"kind": "forced_answer_retry", "finish": f["finish"], "usage": f["usage"]})
        return f["content"], {"mode": "think_reanswer" if finished else "think_forced", "calls": calls,
                              "secs": round(time.time() - t, 1), "reasoning": c["reasoning"], "first_answer": c["content"]}


def seed_of(idx):
    return int(hashlib.md5(str(idx).encode()).hexdigest()[:8], 16)


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def run(a):
    rows = load_rows(a.scores)
    if a.limit:
        vul = [r for r in rows if r["target"] == 1][:a.limit]
        ids = {r["idx"] for r in vul} | {r.get("pair_idx") for r in vul}
        rows = [r for r in rows if r["idx"] in ids]
    os.makedirs(a.out, exist_ok=True)
    cl = Client(a)
    for arm in a.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        path, epath = f"{a.out}/{arm}.jsonl", f"{a.out}/{arm}.errors.jsonl"
        done = {json.loads(l)["idx"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = [r for r in rows if r["idx"] not in done]
        print(f"[{arm}] {len(done)} done, {len(todo)} to do", flush=True)
        fh, eh = open(path, "a", encoding="utf-8"), open(epath, "a", encoding="utf-8")
        T = time.time(); n = n_err = 0; toks = 0

        def work(r):
            messages, meta, thinking = build(r, arm, a.k, hint_style=getattr(a, "hint_style", "neutral"))
            text, rec = cl.judge(messages, thinking, seed_of(r["idx"]))
            js, how = parse_answer(text)
            return {**meta, **rec, "target": r["target"], "pair_idx": r.get("pair_idx"), "status": "ok" if js else "unparsed",
                    "parse": how, "parsed": js, "raw": text}

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
    v = bool(js.get("vulnerable"))
    try:
        p = float(js.get("probability", 50))
    except Exception:
        p = 50.0
    p = min(max(p, 0.0), 100.0)
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
        with open(f"{d}/primevul_test_scores.jsonl", "w", encoding="utf-8") as fh:
            for x in recs:
                modes[x.get("mode", "?")] = modes.get(x.get("mode", "?"), 0) + 1
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
        summary[arm] = {"n": n, "answered": n_ok, "answer_rate": round(n_ok / max(n, 1) * 100, 1), "parsed_leniently": n_len, "modes": modes,
                        "transport_errors": errors, "cut_functions": n_cut,
                        "flag_rate": round(n_vul / max(n_ok, 1) * 100, 1),
                        "verdict_probability_consistent": round(consistent / max(n_ok, 1) * 100, 1),
                        "gen_tokens": {"mean": round(sum(gen) / max(n, 1)), "median": gen[n // 2] if gen else 0, "max": gen[-1] if gen else 0},
                        "lines": {"n_vul_with_flaw_lines": n_fl,
                                  "a_missing_sink_is_a_flaw_line": round(hit_missing / max(n_fl, 1) * 100, 1),
                                  "any_sink_is_a_flaw_line": round(hit_sink / max(n_fl, 1) * 100, 1),
                                  "a_missing_sink_is_a_hint_line": round(hit_hint / max(n_fl, 1) * 100, 1),
                                  "missing_sinks_per_function": round(n_miss_total / max(n_fl, 1), 2)}}
        print(arm, json.dumps(summary[arm]))
    json.dump(summary, open(f"{a.out}/scores/summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=["audit_hint"])
    ap.add_argument("--limit", type=int, default=None, help="smoke: this many vulnerable functions (with flaw lines) + their pairs")
    ap.add_argument("--k", type=int, default=3, help="top-k hint lines (default: 3 to avoid OR-gate false alarm explosion)")
    ap.add_argument("--hint_style", choices=["neutral", "asymmetric"], default="neutral", help="symmetric neutral hints vs asymmetric priming")
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
