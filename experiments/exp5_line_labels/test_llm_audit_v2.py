"""Unit tests for Plan v2 upgrades in llm_audit.py (stdlib only)."""
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_audit as la


def _mask_hint_numbers(text: str) -> str:
    return re.sub(
        r"(flagged these candidate lines for closer inspection: )([0-9, ]+)(\.)",
        r"\1<LINES>\3",
        text,
    )


class LlmAuditV2Test(unittest.TestCase):
    def _sample_row(self):
        # 12 lines: lines 1..4 clustered high score, line 9 also high score, line 0 & 2 braces/comments
        func = "\n".join([
            "void foo(char *src, int n) {",        # 0 (1-based: 1)
            "    char buf[16];",                   # 1 (2)
            "    // comment line",                 # 2 (3)
            "    int len = n;",                    # 3 (4)
            "    int off = 0;",                    # 4 (5)
            "    if (src == NULL)",                # 5 (6)
            "        return;",                     # 6 (7)
            "    off += 1;",                       # 7 (8)
            "    int total = len + off;",          # 8 (9)
            "    memcpy(buf, src, total);",        # 9 (10)
            "    buf[15] = '\\0';",                # 10 (11)
            "}",                                   # 11 (12)
        ])
        scores = [0.1, 0.2, 5.0, 9.0, 8.5, 8.0, 1.0, 0.5, 0.4, 7.5, 0.3, 9.9]
        return {"idx": 12345, "target": 1, "pair_idx": 12346, "func": func, "line_scores": scores}

    def test_prompt_symmetry_plain_hint_nms_vs_plain_rand(self):
        r = self._sample_row()
        m_nms, meta_nms, th_nms = la.build(r, "plain_hint_nms", k=3)
        m_rnd, meta_rnd, th_rnd = la.build(r, "plain_rand", k=3)
        self.assertTrue(th_nms and th_rnd)
        self.assertEqual(m_nms[0], m_rnd[0])
        self.assertEqual(len(meta_nms["hint_lines"]), 3)
        self.assertEqual(len(meta_rnd["hint_lines"]), 3)
        self.assertEqual(_mask_hint_numbers(m_nms[1]["content"]), _mask_hint_numbers(m_rnd[1]["content"]))
        self.assertIn("may already be properly guarded or completely harmless", m_nms[1]["content"])

    def test_prompt_symmetry_debate_nms_vs_debate_rand(self):
        r = self._sample_row()
        m_nms, meta_nms, th_nms = la.build(r, "debate_nms", k=3)
        m_rnd, meta_rnd, th_rnd = la.build(r, "debate_rand", k=3)
        m_non, meta_non, th_non = la.build(r, "debate_none", k=3)
        self.assertTrue(th_nms and th_rnd and th_non)
        self.assertEqual(m_nms[0], m_rnd[0])
        self.assertEqual(_mask_hint_numbers(m_nms[1]["content"]), _mask_hint_numbers(m_rnd[1]["content"]))
        self.assertNotIn("hint_lines", meta_non)
        self.assertNotIn("flagged these candidate lines", m_non[1]["content"])

    def test_nms_hint_lines_spreads_clusters_and_skips_trivial_lines(self):
        r = self._sample_row()
        raw = la.hint_lines(r, "plain_hint", k=3, n_kept=12, nms_radius=0)
        nms = la.hint_lines(r, "plain_hint_nms", k=3, n_kept=12, nms_radius=3)
        # Raw top-3 among is_code lines includes line 12 ('}') and adjacent lines 4, 5
        self.assertIn(12, raw)
        # NMS with mask_trivial excludes '}' (12) and spreads lines 4 and 10
        self.assertNotIn(12, nms)
        self.assertIn(4, nms)
        self.assertIn(10, nms)
        # Check pairwise distance among picked lines (except if fewer than k clusters exist)
        self.assertGreaterEqual(nms[-1] - nms[0], 4)

    def test_to_score_single_and_multi_sample(self):
        # Legacy single-sample
        s, v, p = la.to_score({"vulnerable": True, "probability": 80})
        self.assertTrue(v)
        self.assertAlmostEqual(s, 1.3)
        s0, v0, _ = la.to_score({"vulnerable": False, "probability": 20})
        self.assertFalse(v0)
        self.assertAlmostEqual(s0, -1.3)

        # Multi-sample preserves strict majority sign even at extreme probability
        s_maj_true, vt, _ = la.to_score({"vulnerable": True, "probability": 0, "vote_rate": 2 / 3})
        self.assertTrue(vt)
        self.assertGreater(s_maj_true, 0.0)
        s_maj_false, vf, _ = la.to_score({"vulnerable": False, "probability": 100, "vote_rate": 1 / 3})
        self.assertFalse(vf)
        self.assertLess(s_maj_false, 0.0)

        # Monotonicity across vote rates
        s_all_true, _, _ = la.to_score({"vulnerable": True, "probability": 50, "vote_rate": 1.0})
        s_all_false, _, _ = la.to_score({"vulnerable": False, "probability": 50, "vote_rate": 0.0})
        self.assertGreater(s_all_true, s_maj_true)
        self.assertGreater(s_maj_false, s_all_false)

    def test_parse_defense_json_and_lenient(self):
        js, how = la.parse_defense('Here is my review:\n```json\n{"confirmed": false, "probability": 15, "guard_lines": [6], "reason": "checked"}\n```')
        self.assertEqual(how, "json")
        self.assertFalse(js["confirmed"])
        self.assertEqual(js["probability"], 15)

        js2, how2 = la.parse_defense('confirmed: true, probability: 88')
        self.assertEqual(how2, "lenient")
        self.assertTrue(js2["confirmed"])
        self.assertEqual(js2["probability"], 88.0)

        js3, how3 = la.parse_defense('{"vulnerable": false, "probability": 10}')
        self.assertEqual(how3, "json_vul_fallback")
        self.assertFalse(js3["confirmed"])

    def test_debate_two_stage_orchestration_and_multisample(self):
        r = dict(self._sample_row(), flaw_lines=[9])
        skip_seeds = {la.seed_of(12345) + 1009, la.seed_of(99999) + 1009}

        class FakeClient:
            def __init__(self):
                self.calls = []

            def judge(self, messages, thinking, seed, parser=la.parse_answer):
                self.calls.append((len(messages), messages[0]["content"][:20], seed))
                if parser in (la.parse_answer, la.parse_answer_json):
                    # Sample index 1 flags vulnerable=False (skips defense); samples 0 and 2 flag True
                    if seed in skip_seeds:
                        return '{"vulnerable": false, "probability": 20, "claim": null}', {
                            "mode": "think_complete", "calls": [{"usage": {"completion_tokens": 50}}], "secs": 0.5, "reasoning": "ok"
                        }
                    return '{"vulnerable": true, "probability": 80, "claim": {"line": 10, "cwe": "CWE-120", "variable": "total", "trigger": "total > 16"}}', {
                        "mode": "think_complete", "calls": [{"usage": {"completion_tokens": 60}}], "secs": 0.6, "reasoning": "pros"
                    }
                # Defense rejects the claim!
                return '{"confirmed": false, "probability": 10, "guard_lines": [6], "reason": "bounded"}', {
                    "mode": "think_complete", "calls": [{"usage": {"completion_tokens": 40}}], "secs": 0.4, "reasoning": "def"
                }

        cl = FakeClient()
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "scores.jsonl")
            sub_path = os.path.join(td, "subset.json")
            out_dir = os.path.join(td, "out")
            with open(in_path, "w") as fh:
                fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps({**r, "idx": 99999, "target": 0, "pair_idx": 12345}) + "\n")
            with open(sub_path, "w") as fh:
                json.dump({"pair_indices": [[12345, 99999]]}, fh)

            class Args:
                scores = in_path
                out = out_dir
                arms = ["debate_nms"]
                limit = None
                subset_json = sub_path
                k = 3
                nms_radius = 3
                n_samples = 3
                hint_style = "neutral"
                system_override = None
                plain_template_file = None
                workers = 2
                url = "http://localhost:8000/v1"
                model = "llm"
                think_budget = 1024
                answer_tokens = 256
                max_model_len = 8192
                timeout = 60
                retries = 1
                tokenizer = "none"

            # Monkeypatch Client inside llm_audit.run
            orig_client = la.Client
            la.Client = lambda a: cl
            try:
                la.run(Args())
                la.score(Args())
            finally:
                la.Client = orig_client

            summ = json.load(open(os.path.join(out_dir, "scores", "summary.json")))["debate_nms"]
            self.assertEqual(summ["n"], 2)
            self.assertEqual(summ["n_samples"], 3)
            self.assertEqual(summ["flag_rate"], 0.0)
            self.assertEqual(summ["debate_stats"]["defense_called"], 4)  # 2 per fn where prosecutor flagged
            self.assertEqual(summ["debate_stats"]["defense_rejected_rate"], 100.0)
            self.assertEqual(summ["lines"]["any_sink_is_a_flaw_line"], 100.0)

    def test_subset_ids_loading(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sub.json")
            with open(p, "w") as fh:
                json.dump({"clean_pair_indices": [[1, 2], [3, 4]]}, fh)
            self.assertEqual(la._load_subset_ids(p), {1, 2, 3, 4})

    def test_parse_answer_openvul_tags_and_template_symmetry(self):
        js1, h1 = la.parse_answer("Analysis complete.\n\nNO_VUL\n")
        self.assertEqual(h1, "tag")
        self.assertFalse(js1["vulnerable"])
        js2, h2 = la.parse_answer("Earlier note NO_VUL replaced after check.\nHAS_VUL")
        self.assertEqual(h2, "tag")
        self.assertTrue(js2["vulnerable"])

        tpl_path = os.path.join(os.path.dirname(__file__), "inputs", "openvul_grpo_plain_templates.json")
        tpls = la._load_plain_templates(tpl_path)
        r = self._sample_row()
        m_nms, _, _ = la.build(r, "plain_hint_nms", k=3, plain_templates=tpls)
        m_rnd, _, _ = la.build(r, "plain_rand", k=3, plain_templates=tpls)
        self.assertEqual(m_nms[0], m_rnd[0])
        self.assertIn("HAS_VUL", m_nms[1]["content"])
        self.assertEqual(_mask_hint_numbers(m_nms[1]["content"]), _mask_hint_numbers(m_rnd[1]["content"]))


if __name__ == "__main__":
    unittest.main()
