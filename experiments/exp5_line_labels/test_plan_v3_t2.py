"""Pure-stdlib unit tests for Plan v3 Tier 2 Paired Anti-Shortcut QLoRA helpers."""
import ast
import json
import math
import os
import random
import sys
import tempfile
import unittest

EXP5 = os.path.dirname(os.path.abspath(__file__))
if EXP5 not in sys.path:
    sys.path.insert(0, EXP5)

import train_lora_t2 as t2  # noqa: E402


class PlanV3T2Test(unittest.TestCase):
    def test_sample_length_matched_benigns_matches_skewed_distribution_and_sparse_tail(self):
        rng = random.Random(7)
        # Skewed target pairs around ~2000-4000 chars
        train_pairs = [
            {"cv": 1500 + (i % 5) * 300, "cf": 1800 + (i % 7) * 350}
            for i in range(60)
        ]
        # Benign pool dominated by short functions (100-400 chars), plus sparse long ones (1600-3800)
        pool = []
        for i in range(300):
            c = 120 + (i % 20) * 15
            pool.append({"idx": i, "func": "x" * c, "chars": c})
        for j in range(90):
            c = 1550 + j * 25
            pool.append({"idx": 1000 + j, "func": "x" * c, "chars": c})

        sel, stats = t2.sample_length_matched_benigns(
            train_pairs, pool, n_total=100, matched_frac=0.80, n_bins=10, seed=42,
        )
        self.assertEqual(len(sel), 100)
        self.assertEqual(len({r["idx"] for r in sel}), 100)
        self.assertEqual(stats["matched"], 80)
        self.assertEqual(stats["random"], 20)
        # Median should be pulled strongly toward the longer pair regime (>1500 vs pool's ~280)
        self.assertGreater(stats["benign_median_chars"], 1400)

        # Sparse edge case: request more than pool or sparse single extreme bin -> still zero dupes
        sel_sparse, st_sp = t2.sample_length_matched_benigns(
            train_pairs, pool[:25], n_total=50, matched_frac=0.80, n_bins=20, seed=0,
        )
        self.assertEqual(len(sel_sparse), 25)
        self.assertEqual(len({r["idx"] for r in sel_sparse}), 25)
        self.assertEqual(st_sp["selected"], 25)

    def test_compute_pair_length_weights_balances_mass_and_clips(self):
        # 18 add pairs (cf > cv), 2 rem pairs (cv > cf), 1 equal pair -> rare weights hit max_weight=4.0 clip
        pairs = (
            [{"cv": 100, "cf": 150} for _ in range(18)]
            + [{"cv": 200, "cf": 120} for _ in range(2)]
            + [{"cv": 100, "cf": 100}]
        )
        w = t2.compute_pair_length_weights(pairs, max_weight=4.0)
        self.assertEqual(len(w), len(pairs))
        self.assertTrue(all(0.1 <= wi <= 4.0 for wi in w))
        self.assertEqual(w[18], 4.0)
        self.assertEqual(w[19], 4.0)

        # Realistic ~PrimeVul ratio (10:2 = 5:1, within clip): exact mean=1.0 and 50/50 add/rem mass
        pairs_bal = [{"cv": 100, "cf": 150} for _ in range(10)] + [{"cv": 200, "cf": 120} for _ in range(2)]
        wb = t2.compute_pair_length_weights(pairs_bal, max_weight=4.0)
        self.assertAlmostEqual(sum(wb) / len(wb), 1.0, places=6)
        self.assertAlmostEqual(sum(wb[:10]), sum(wb[10:]), places=6)

    def test_load_valid_extra_benigns_excludes_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid_pairs = [{"v_idx": 1, "f_idx": 2, "commit_id": "cv1", "v_func": "v1", "f_func": "f1"}]
            test_pairs = [{"v_idx": 10, "f_idx": 11, "commit_id": "ct1", "v_func": "vt1", "f_func": "void ft() { return; }"}]
            rows_test = [{"idx": 10, "target": 1, "commit_id": "ct1", "func": "vt1"},
                         {"idx": 11, "target": 0, "commit_id": "ct1", "func": "void ft() { return; }"}]
            rows_valid = [
                {"idx": 2, "target": 0, "commit_id": "cv1", "func": "f1"},  # already in valid_pairs
                {"idx": 20, "target": 0, "commit_id": "ct1", "func": "other"},  # commit leaked into test
                {"idx": 21, "target": 0, "commit_id": "c_ok1", "func": "void ft(){return;}"},  # text_hash leaked into test
                {"idx": 22, "target": 1, "commit_id": "c_ok2", "func": "vul"},  # target != 0
                {"idx": 23, "target": 0, "commit_id": "c_clean", "project": "p1", "func": "int safe() { return 1; }"},
            ]
            for fn, rs in (("primevul_test.jsonl", rows_test), ("primevul_valid.jsonl", rows_valid)):
                with open(os.path.join(tmp, fn), "w", encoding="utf-8") as fh:
                    for r in rs:
                        fh.write(json.dumps(r) + "\n")

            got = t2.load_valid_extra_benigns(tmp, valid_pairs, test_pairs, n_sample=10, seed=0)
            self.assertEqual([r["idx"] for r in got], [23])

    def test_build_manual40_and_taxonomy_masks_real_files_exact_40_match(self):
        tax_path = os.path.join(EXP5, "results", "patch_taxonomy", "test_pairs.jsonl")
        man40_path = os.path.join(EXP5, "results", "patch_taxonomy", "manual_labels_test40.json")
        tax_rows = t2.load_jsonl(tax_path)
        fake_test_pairs = [{"v_idx": r["idx"], "project": r["project"]} for r in tax_rows]

        masks, info = t2.build_manual40_and_taxonomy_masks(
            fake_test_pairs, manual40_path=man40_path, taxonomy_path=tax_path,
        )
        self.assertEqual(info["manual40_matched"], 40)
        self.assertEqual(info["taxonomy_matched"], 564)
        self.assertEqual(sum(masks["manual40_local"]), 15)
        self.assertEqual(sum(masks["manual40_outside"]), 14)
        self.assertEqual(sum(masks["manual40_not_a_to_b"]), 11)
        self.assertEqual(sum(masks["tax_local_ab"]), 58)
        self.assertEqual(sum(masks["tax_out_ab"]), 296)
        self.assertEqual(sum(masks["tax_not_a_to_b"]), 210)

    def test_composite_val_gate_and_report_formatting(self):
        self.assertAlmostEqual(t2.compute_t2_composite_val(75.0, 0.71), 0.73, places=6)
        dummy_pairs = {
            "all": {"rank_acc": 80.0, "rank_acc_debiased": 79.0, "ties_pct": 0.2, "spearman_log_chars": 0.11,
                    "t0": {"net": 15.2}, "t_median": {"net": 16.0}},
            "length_matched_balanced": {"rank_acc": 74.0},
            "length_matched_5pct": {"rank_acc": 75.0},
            "patch_removes_code": {"rank_acc": 63.1},
            "seen_projects": {"rank_acc": 80.5},
            "unseen_projects": {"rank_acc": 78.0},
            "changed": {"rank_acc": 79.0},
            "add_only": {"rank_acc": 81.5},
        }
        dummy_full = {
            "roc_auc": 0.72, "debiased_roc_auc": 0.69, "auprc": 0.12,
            "vds_fnr_at_05pct_fpr": 88.0, "recall_at_05pct_fpr": 12.0,
            "actual_fpr_pct": 0.48, "spearman_log_chars": 0.15,
        }
        gate = t2.evaluate_t2_gate(dummy_pairs, dummy_full)
        self.assertEqual(gate["verdict"], "PASS_T2_GATE")
        md = t2.format_t2_report_md({
            "telemetry": {"model": "Q7B", "lora_r": 16, "lora_alpha": 32, "mid_layer": 13, "last_layer": 27, "max_len": 2048, "trainable_params": 40, "peak_vram_gb": 17.5, "train_secs": 100.0},
            "split_stats": {"train_pairs": 4347, "train_add": 3574, "train_rem": 637, "valid_pairs": 562, "valid_benign": 1000},
            "benign_stats": {"selected": 4347, "matched": 3478, "random": 869, "pair_median_chars": 2100, "benign_median_chars": 2050},
            "curve": [{"epoch": 0.5, "pairs_done": 2174, "train_loss": 0.91, "val_bal_rank": 73.0, "val_rem_rank": 62.0, "val_debias_roc": 0.66, "val_net_med": 12.0, "val_composite": 0.695}],
            "test_pairs": dummy_pairs,
            "test_fulltest": dummy_full,
            "gate": gate,
        })
        self.assertIn("PASS_T2_GATE", md)
        self.assertIn("lora_nohint", md)

    def test_pair_benigns_by_length_aligns_and_handles_smaller_pool(self):
        pair_lens = [100, 900, 200, 1500, 400, 1800, 300, 1100]
        benign_lens = [1810, 105, 410, 915, 195, 1490, 290, 1095]
        matched = t2.pair_benigns_by_length(pair_lens, benign_lens, window=2, seed=42)
        self.assertEqual(len(matched), len(pair_lens))
        self.assertEqual(set(matched), set(range(len(benign_lens))))
        # Even with window=2, shortest pair (100) should map close to shortest benigns (<300), longest (1800) to longest (>1400)
        self.assertLess(benign_lens[matched[0]], 300)
        self.assertGreater(benign_lens[matched[5]], 1400)

        # M < N cyclic length alignment should cover all M benigns and preserve ordering roughly
        small_b = [120, 1000, 1750]
        matched_sm = t2.pair_benigns_by_length(pair_lens, small_b, window=1, seed=0)
        self.assertEqual(len(matched_sm), len(pair_lens))
        self.assertEqual(set(matched_sm), {0, 1, 2})
        self.assertEqual(matched_sm[0], 0)
        self.assertEqual(matched_sm[5], 2)

    def test_build_length_bucketed_batches_packs_and_covers_all(self):
        # 6 short (<=512), 3 mid (513..1024), 2 long (>1024)
        items = [
            (0, 100), (1, 300), (2, 512), (3, 250), (4, 480), (5, 511),
            (6, 513), (7, 800), (8, 1024),
            (9, 1025), (10, 2048),
        ]
        b1 = t2.build_length_bucketed_batches(items, seed=42)
        b2 = t2.build_length_bucketed_batches(items, seed=42)
        self.assertEqual(b1, b2)
        flat = [pi for mb in b1 for pi in mb]
        self.assertEqual(sorted(flat), list(range(len(items))))
        len_map = dict(items)
        for mb in b1:
            max_l = max(len_map[pi] for pi in mb)
            if max_l <= 512:
                self.assertLessEqual(len(mb), 4)
            elif max_l <= 1024:
                self.assertLessEqual(len(mb), 2)
            else:
                self.assertEqual(len(mb), 1)

    def test_t2_scripts_parse_cleanly(self):
        for rel in ("train_lora_t2.py", os.path.join("colab", "presets.py")):
            with open(os.path.join(EXP5, rel), encoding="utf-8") as fh:
                ast.parse(fh.read())


if __name__ == "__main__":
    unittest.main()
