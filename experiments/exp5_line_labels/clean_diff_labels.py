"""Clean diff-aligned line labels, variable-collar masking, and NMS hunk ranking (CPU, stdlib only).

Addresses three empirically verified Tier 1 label/ranking defects:
1. Duplicate-stealing & compound-block spillover in SecVulEval `flat.find(k)` statement matching:
   when a paired fixed function exists, intersect/align statement matches with true `difflib`
   changed/deleted lines and filter trivial lines (`{`, `}`, blank, comment, bare `else`).
2. Variable-collar conflict in `listwise` / `BCE` training: adjacent lines (±radius) sharing
   variables with a flaw line (e.g. an `if` guard right above a flaw sink, or vice versa) are
   returned as collar lines so training can mask them (`y = -1`) instead of forcing the model
   to suppress the partner of the same bug.
3. Top-5 region clustering: `diverse_topk_order` suppresses trivial lines and applies greedy
   1D NMS (radius `r=3`) so Top-k probes distinct candidate regions instead of stacked lines.
"""
from __future__ import annotations

import bisect
import difflib
import re

from label_noise import line_kinds

KEYWORDS = frozenset(
    """auto break case char const continue default do double else enum extern float for goto if inline int long
    register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
    true false NULL nullptr bool class new delete this public private protected template typename namespace using
    virtual override try catch throw size_t ssize_t uintptr_t intptr_t uint8_t uint16_t uint32_t uint64_t
    int8_t int16_t int32_t int64_t u8 u16 u32 u64 s8 s16 s32 s64 gboolean gint guint gsize""".split()
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_COND_RE = re.compile(r"^\s*(\}?\s*else\s+)?(if|while|for|switch)\b")


def norm_ws(text: str) -> str:
    """Strip all whitespace for whitespace-insensitive statement matching."""
    return re.sub(r"\s+", "", text or "")


def line_idents(line: str) -> set[str]:
    """Return non-keyword C/C++ variable identifiers in `line` (excluding function call names)."""
    out = set()
    for m in _IDENT_RE.finditer(line):
        w = m.group(0)
        if w in KEYWORDS or line[m.end():m.end() + 1].lstrip().startswith("("):
            continue
        out.add(w)
    return out


def diff_changed_lines(vul_func: str, fix_func: str) -> tuple[set[int], list[tuple[int, int, int]]]:
    """Return `(changed_in_vul, insert_points)` from a stripped-line SequenceMatcher diff."""
    a = [x.strip() for x in vul_func.split("\n")]
    b = [x.strip() for x in fix_func.split("\n")]
    changed: set[int] = set()
    inserts: list[tuple[int, int, int]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            changed.update(range(i1, i2))
        elif op == "insert":
            inserts.append((i1, j1, j2))
    return changed, inserts


def diff_flaw_lines_clean(vul_func: str, fix_func: str, drop_trivial: bool = True) -> list[int]:
    """Exact non-trivial changed/deleted lines of `vul_func` relative to `fix_func`."""
    changed, _ = diff_changed_lines(vul_func, fix_func)
    if not changed:
        return []
    if not drop_trivial:
        lines = vul_func.split("\n")
        return sorted(i for i in changed if lines[i].strip())
    kinds = line_kinds(vul_func)
    return sorted(i for i in changed if i < len(kinds) and kinds[i] is None)


def _all_statement_spans(func: str, stmts: list[str]) -> list[list[int]]:
    """Find every line-index span in `func` matching each statement in `stmts` (whitespace-stripped)."""
    flat_chars, pos = [], []
    for i, ch in enumerate(func):
        if not ch.isspace():
            flat_chars.append(ch)
            pos.append(i)
    flat = "".join(flat_chars)
    starts = [0] + [i + 1 for i, ch in enumerate(func) if ch == "\n"]
    spans_out: list[list[int]] = []
    for s in stmts:
        k = norm_ws(s)
        if not k:
            continue
        occ = []
        start = 0
        while True:
            p = flat.find(k, start)
            if p < 0:
                break
            a, b = pos[p], pos[p + len(k) - 1]
            la = bisect.bisect_right(starts, a) - 1
            lb = bisect.bisect_right(starts, b) - 1
            occ.append(list(range(la, lb + 1)))
            start = p + 1
        if occ:
            spans_out.append(occ)
    return spans_out


def secvul_flaw_lines_fixed(
    vul_func: str,
    stmts: list[str],
    fix_func: str | None = None,
    drop_trivial: bool = True,
) -> tuple[list[int], int]:
    """Locate `stmts` in `vul_func` without early-duplicate stealing or unchanged-block spillover.

    When `fix_func` is provided and has changed/deleted lines, statement matches are preferred
    where they overlap the true `difflib` changed set and intersected with it so unchanged lines
    inside compound AST statements (e.g. moved `if`/`while` blocks) are not mislabelled as flaws.
    Returns `(sorted_line_indices, missed_statement_count)`.
    """
    kinds = line_kinds(vul_func)
    all_occ = _all_statement_spans(vul_func, stmts)
    missed = sum(1 for s in stmts if norm_ws(s)) - len(all_occ)
    changed_non_triv: set[int] = set()
    if fix_func is not None:
        raw_changed, _ = diff_changed_lines(vul_func, fix_func)
        changed_non_triv = {
            i for i in raw_changed if i < len(kinds) and (not drop_trivial or kinds[i] is None)
        }

    picked: set[int] = set()
    cursor = -1
    for occ_list in all_occ:
        if changed_non_triv:
            overlapping = [sp for sp in occ_list if any(i in changed_non_triv for i in sp)]
            if overlapping:
                for sp in overlapping:
                    picked.update(i for i in sp if i in changed_non_triv)
                continue
        # No diff overlap available (unpaired function or add-only patch): prefer span at/after cursor
        chosen = next((sp for sp in occ_list if sp[0] >= cursor), occ_list[0])
        cursor = chosen[-1] + 1
        picked.update(chosen)

    if drop_trivial:
        picked = {i for i in picked if i < len(kinds) and kinds[i] is None}
    return sorted(picked), missed


def variable_collar_lines(func: str, flaw_lines: list[int], radius: int = 2) -> list[int]:
    """Return non-flaw code lines within `±radius` of `flaw_lines` that belong to the same
    Guard/Sink chain (sharing an identifier or immediately adjacent control condition).

    Used during training (`y = -1`) so listwise/BCE losses do not force the model to penalize
    the adjacent Guard or Sink partner of the labelled flaw line.
    """
    if radius <= 0 or not flaw_lines:
        return []
    lines = func.split("\n")
    kinds = line_kinds(func)
    flaw_set = {i for i in flaw_lines if 0 <= i < len(lines)}
    if not flaw_set:
        return []
    idents_map = {i: line_idents(lines[i]) for i in range(len(lines)) if kinds[i] is None}
    collar: set[int] = set()
    for f in flaw_set:
        if f not in idents_map:
            continue
        fv = idents_map[f]
        f_cond = bool(_COND_RE.match(lines[f]))
        for j in range(max(0, f - radius), min(len(lines), f + radius + 1)):
            if j in flaw_set or j not in idents_map:
                continue
            jv = idents_map[j]
            if fv & jv:
                collar.add(j)
            elif abs(j - f) == 1 and (f_cond or _COND_RE.match(lines[j])):
                collar.add(j)
    return sorted(collar)


def diverse_topk_order(
    func: str,
    scores: list[float],
    radius: int = 3,
    mask_trivial: bool = True,
) -> list[int]:
    """Order line indices for hunk-level recall: push trivial lines down, then apply greedy 1D NMS.

    Each picked code line suppresses neighbors within `±radius` lines to the back of the queue
    so the top-k lines cover distinct code regions instead of clustering on one hunk.
    """
    n = min(len(scores), func.count("\n") + 1)
    base = sorted(range(n), key=lambda i: (-scores[i], i))
    if mask_trivial:
        kinds = line_kinds(func)
        code = [i for i in base if i < len(kinds) and kinds[i] is None]
        triv = [i for i in base if i >= len(kinds) or kinds[i] is not None]
    else:
        code, triv = base, []

    if radius <= 0:
        return code + triv

    picked, deferred = [], []
    for i in code:
        if any(abs(i - p) <= radius for p in picked):
            deferred.append(i)
        else:
            picked.append(i)
    return picked + deferred + triv
