"""Pure-stdlib unit tests for Plan v3 Phase 3.0 audit and Tier 1 probe helpers."""
import ast
import json
import os
import sys
import tempfile
import unittest

EXP5 = os.path.dirname(os.path.abspath(__file__))
if EXP5 not in sys.path:
    sys.path.insert(0, EXP5)

import probe_llm_t1 as t1  # noqa: E402


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        toks = [abs(hash(tok)) % 10000 for tok in text.split()] if text else []
        if add_special_tokens:
            return [1] + toks
        return toks


class PlanV3T1Test(unittest.TestCase):
    def test_build_probe_input_ids_preserves_suffix_on_truncation(self):
        tok = FakeTok()
        long_code = "\n".join(f"int var_{i} = {i};" for i in range(600))
        ids, cut = t1.build_probe_input_ids(long_code, tok, max_len=128)
        self.assertTrue(cut)
        self.assertEqual(len(ids), 128)
        suf = tok.encode(t1.PROBE_SUFFIX, add_special_tokens=False)
        self.assertEqual(ids[-len(suf):], suf)

        prompt_str, cut_str = t1.build_probe_prompt(long_code, max_chars=350)
        self.assertTrue(cut_str)
        self.assertTrue(prompt_str.endswith(t1.PROBE_SUFFIX))

    def test_make_clean_splits_filters_commit_hash_and_marks_seen_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            def _write_jsonl(name, rows):
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")

            test_pair = [
                {"idx": 100, "target": 1, "commit_id": "c_test1", "project": "seen_proj", "func": "void a() { int x = 1; }"},
                {"idx": 101, "target": 0, "commit_id": "c_test1", "project": "seen_proj", "func": "void a() { if (1) return; int x = 1; }"},
                {"idx": 102, "target": 1, "commit_id": "c_test2", "project": "unseen_proj", "func": "void b() { strcpy(p, q); }"},
                {"idx": 103, "target": 0, "commit_id": "c_test2", "project": "unseen_proj", "func": "void b() { strncpy(p, q, 4); }"},
            ]
            # Train has 1 clean pair in seen_proj, 1 leaked by commit_id, 1 leaked by whitespace-normalized hash
            train_pair = [
                {"idx": 1, "target": 1, "commit_id": "c_tr1", "project": "seen_proj", "func": "void ok() { *p = 0; }"},
                {"idx": 2, "target": 0, "commit_id": "c_tr1", "project": "seen_proj", "func": "void ok() { if (p) *p = 0; }"},
                {"idx": 3, "target": 1, "commit_id": "c_test1", "project": "seen_proj", "func": "void diff() { int y = 2; }"},
                {"idx": 4, "target": 0, "commit_id": "c_test1", "project": "seen_proj", "func": "void diff() { int y = 3; }"},
                {"idx": 5, "target": 1, "commit_id": "c_tr_hash", "project": "other_proj", "func": "void b() {\n  strcpy(p,q);\n}"},
                {"idx": 6, "target": 0, "commit_id": "c_tr_hash", "project": "other_proj", "func": "void b() { return; }"},
            ]
            valid_pair = [
                {"idx": 10, "target": 1, "commit_id": "c_va1", "project": "other_proj", "func": "void v1() { a[i] = 1; }"},
                {"idx": 11, "target": 0, "commit_id": "c_va1", "project": "other_proj", "func": "void v1() { if (i < n) a[i] = 1; }"},
            ]
            # Train full includes 1 leaked benign (same hash as test), 1 clean benign
            train_full = train_pair + [
                {"idx": 50, "target": 0, "commit_id": "c_leaked_full", "project": "p", "func": "void a(){if(1)return;int x=1;}"},
                {"idx": 51, "target": 0, "commit_id": "c_clean_eb", "project": "p", "func": "int safe_fn() { return 0; }"},
            ]
            test_full = test_pair + [
                {"idx": 200, "target": 1, "commit_id": "c_rest_v", "project": "unseen_proj", "func": "void rv() {}"},
                {"idx": 201, "target": 0, "commit_id": "c_rest_b", "project": "unseen_proj", "func": "void rb() {}"},
            ]
            _write_jsonl("primevul_train_paired.jsonl", train_pair)
            _write_jsonl("primevul_valid_paired.jsonl", valid_pair)
            _write_jsonl("primevul_test_paired.jsonl", test_pair)
            _write_jsonl("primevul_train.jsonl", train_full)
            _write_jsonl("primevul_test.jsonl", test_full)

            man = {
                "n_full": 6, "n_full_vulnerable": 3,
                "strata": {"P": {"weight": 1.0}, "V": {"weight": 1.0}, "B": {"weight": 10.0}},
                "rest_vulnerable_idx": [200], "sampled_benign_idx": [201],
            }
            man_path = os.path.join(tmp, "manifest.json")
            with open(man_path, "w", encoding="utf-8") as fh:
                json.dump(man, fh)

            sp = t1.make_clean_splits(tmp, manifest_path=man_path, extra_benign=10, seed=0)
            self.assertEqual(len(sp["train_pairs"]), 1)
            self.assertEqual(sp["train_pairs"][0]["v_idx"], 1)
            self.assertEqual(sp["stats"]["train_filter"]["dropped_commit"], 1)
            self.assertEqual(sp["stats"]["train_filter"]["dropped_hash"], 1)
            self.assertEqual([r["idx"] for r in sp["extra_train_benign"]], [51])
            self.assertEqual(sp["stats"]["test_seen_projects"], 1)
            self.assertEqual(sp["stats"]["test_unseen_projects"], 1)
            self.assertEqual(len(sp["fulltest_rows"]), 6)

    def test_scripts_parse_cleanly(self):
        for fname in ("probe_llm_t1.py", "audit_gd3_cpu.py"):
            src = open(os.path.join(EXP5, fname), encoding="utf-8").read()
            ast.parse(src)


if __name__ == "__main__":
    unittest.main()
