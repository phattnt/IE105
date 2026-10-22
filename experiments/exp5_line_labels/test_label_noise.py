"""Unit tests for the trivial flaw-line rule and the filter (python -m unittest test_label_noise)."""
import unittest

from label_noise import drop_trivial_flaws, filter_labels, is_trivial_line, line_kinds


class TrivialLine(unittest.TestCase):
    def test_trivial(self):
        for line, kind in [("}", "brace"), ("{", "brace"), ("  } else {", "else"), ("else", "else"), ("};", "brace"),
                           ("\t})", "brace"), ("/* ... */", "comment"), ("// note", "comment"), (";", "punct"),
                           ("  );", "punct"), ("", "blank"), ("   \t", "blank"), ("} /* end if */", "brace")]:
            with self.subTest(line=line):
                self.assertTrue(is_trivial_line(line))
                self.assertEqual(line_kinds(line)[0], kind)

    def test_not_trivial(self):
        for line in ["return -1;", "free(p);", "i++;", "break;", "continue;", "return;", "} else if (x) {", "*p = 0;",
                     "x = '{';", 'puts("}");', "goto out;", "out:", "do {", "} while (0);", "#endif", "a[i] = 0; // fix"]:
            with self.subTest(line=line):
                self.assertFalse(is_trivial_line(line))

    def test_block_comment_spans_lines(self):
        func = "int f(int *p)\n{\n\t/*\n\t * check p\n\t */\n\t*p = 0;\n\treturn 0;\n}"
        self.assertEqual(line_kinds(func), [None, "brace", "comment", "comment", "comment", None, None, "brace"])

    def test_comment_markers_inside_strings(self):
        func = 'a = "/*";\n}\nb = 1; /* x */'
        self.assertEqual(line_kinds(func), [None, "brace", None])


class Filter(unittest.TestCase):
    def rec(self, idx, target, func, flaws, pair=None):
        return {"idx": idx, "target": target, "func": func, "n_lines": func.count("\n") + 1, "flaw_lines": flaws, "pair_idx": pair}

    def test_drop_trivial_keeps_lines(self):
        r = self.rec(1, 1, "void f()\n{\n\tfree(p);\n}\n", [1, 2, 3, 4])
        kept, removed = drop_trivial_flaws(r)
        self.assertEqual(kept, [2])
        self.assertEqual(removed, {1: "brace", 3: "brace", 4: "blank"})
        out, st = filter_labels([r], drop_trivial=True)
        self.assertEqual(out[0]["func"], r["func"])
        self.assertEqual(out[0]["flaw_lines"], [2])
        self.assertEqual(r["flaw_lines"], [1, 2, 3, 4])           # input record not mutated

    def test_all_trivial_becomes_unlabelled(self):
        r = self.rec(1, 1, "void f()\n{\n}", [1, 2])
        out, st = filter_labels([r], drop_trivial=True)
        self.assertEqual(out[0]["flaw_lines"], [])
        self.assertEqual(st["vulnerable_left_without_lines"], 1)

    def test_density_drops_pair(self):
        dense = self.rec(1, 1, "a;\nb;\nc;", [0, 1], pair=2)
        fixed = self.rec(2, 0, "a;\nb2;\nc;", [], pair=1)
        ok = self.rec(3, 1, "a;\nb;\nc;", [1], pair=4)
        fixed2 = self.rec(4, 0, "a;\nb2;\nc;", [], pair=3)
        out, st = filter_labels([dense, fixed, ok, fixed2], max_density=0.5)
        self.assertEqual([x["idx"] for x in out], [3, 4])
        self.assertEqual(st["functions_dropped_by_density"], 2)

    def test_off_by_default(self):
        r = self.rec(1, 1, "{\n}", [0, 1])
        out, st = filter_labels([r])
        self.assertEqual(out, [r])


if __name__ == "__main__":
    unittest.main()
