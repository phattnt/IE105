import unittest

from clean_diff_labels import (
    diff_flaw_lines_clean,
    diverse_topk_order,
    secvul_flaw_lines_fixed,
    variable_collar_lines,
)


class CleanDiffLabelsTest(unittest.TestCase):
    def test_prevents_early_duplicate_stealing(self):
        # Line 2 has an unchanged "return -1;"; line 6 has the changed "return -1;".
        vul = "\n".join([
            "int f(int a, int b) {",
            "  if (a < 0)",
            "    return -1;",
            "  int x = a + b;",
            "  if (b > 10)",
            "    return -1;",
            "  return x;",
            "}",
        ])
        fix = "\n".join([
            "int f(int a, int b) {",
            "  if (a < 0)",
            "    return -1;",
            "  int x = a + b;",
            "  if (b > 10)",
            "    return -2;",
            "  return x;",
            "}",
        ])
        fl, missed = secvul_flaw_lines_fixed(vul, ["return -1;"], fix_func=fix, drop_trivial=True)
        self.assertEqual(missed, 0)
        self.assertEqual(fl, [5])

    def test_trims_compound_block_spillover_to_true_diff(self):
        # SecVulEval changed_statements sometimes gives an outer `if (...) { ... }` block
        # when only one inner statement changed in the patch.
        vul = "\n".join([
            "void g(char *dst, const char *src, int n) {",
            "  if (n > 0) {",
            "    int tmp = n;",
            "    strcpy(dst, src);",
            "    log_done(tmp);",
            "  }",
            "}",
        ])
        fix = "\n".join([
            "void g(char *dst, const char *src, int n) {",
            "  if (n > 0) {",
            "    int tmp = n;",
            "    strncpy(dst, src, n);",
            "    log_done(tmp);",
            "  }",
            "}",
        ])
        stmt = "if (n > 0) {\n    int tmp = n;\n    strcpy(dst, src);\n    log_done(tmp);\n  }"
        fl, _ = secvul_flaw_lines_fixed(vul, [stmt], fix_func=fix, drop_trivial=True)
        self.assertEqual(fl, [3])

    def test_diff_flaw_lines_clean_drops_trivial_comments_and_braces(self):
        vul = "\n".join([
            "void h(int *p) {",
            "  /* old comment",
            "     still comment */",
            "  {",
            "    *p = 1;",
            "  }",
            "}",
        ])
        fix = "\n".join([
            "void h(int *p) {",
            "  if (!p) return;",
            "  *p = 2;",
            "}",
        ])
        self.assertEqual(diff_flaw_lines_clean(vul, fix, drop_trivial=True), [4])

    def test_variable_collar_lines_finds_guard_sink_partner(self):
        func = "\n".join([
            "void k(char *buf, int len, int unrelated) {",  # 0
            "  unrelated++;",                              # 1 (no shared var with flaw)
            "  if (len > 0) {",                            # 2 (shares len with flaw line 3)
            "    pop_n(len * 2);",                         # 3 <- flaw line
            "  }",                                         # 4 (trivial brace)
            "  use_other(unrelated);",                     # 5
            "}",                                           # 6
        ])
        self.assertEqual(variable_collar_lines(func, [3], radius=2), [2])

    def test_diverse_topk_order_masks_trivial_and_spreads_regions(self):
        func = "\n".join([
            "int f(int a) {",    # 0
            "  if (a > 0) {",    # 1 (score 9.0)
            "    foo(a);",       # 2 (score 8.5, within radius 2 of line 1)
            "    bar(a);",       # 3 (score 8.0, within radius 2 of line 1)
            "  }",               # 4 (score 9.5, trivial -> pushed to bottom)
            "  baz(a);",         # 5 (score 6.0, outside radius 2 of line 1)
            "}",                 # 6
        ])
        scores = [1.0, 9.0, 8.5, 8.0, 9.5, 6.0, 0.5]
        order = diverse_topk_order(func, scores, radius=2, mask_trivial=True)
        self.assertEqual(order[:2], [1, 5])
        self.assertEqual(set(order[-2:]), {4, 6})


if __name__ == "__main__":
    unittest.main()
