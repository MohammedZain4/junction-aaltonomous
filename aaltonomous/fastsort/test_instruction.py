"""python -m unittest aaltonomous.fastsort.test_instruction  (stdlib only)"""
import ast
import itertools
import pathlib
import unittest

from aaltonomous.fastsort.instruction import parse_instruction

TASKS = pathlib.Path(__file__).resolve().parents[1] / "interface" / "tasks.py"


def _templates():
    tree = ast.parse(TASKS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TEMPLATES" for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "FIXED_INSTRUCTION" for t in node.targets):
            pass
    raise RuntimeError("TEMPLATES not found")


def M(r, y, b):
    return {"red": r, "yellow": y, "blue": b}


# unseen paraphrases -> expected colour mapping
PARAPHRASES = [
    ("Put the red objects in the left bin, the yellow objects in the center bin and the blue objects in the right bin.",
     M("left", "center", "right")),
    ("Sort the objects: red into right, blue into left, yellow into center.", M("right", "center", "left")),
    ("Red stuff goes on the right, blue stuff on the left, and yellow in the middle.", M("right", "center", "left")),
    ("Put the red, yellow and blue objects in the left, center and right bins respectively.",
     M("left", "center", "right")),
    ("Blue, red and yellow go into the right, center and left bins, respectively.", M("center", "left", "right")),
    ("Everything blue belongs in the centre box; red pieces go to the leftmost box and yellow to the rightmost.",
     M("left", "right", "center")),
    ("Could you please place the yellow items into the bin on the left? The red ones should go in the middle one, "
     "and the blue ones on the right.", M("center", "left", "right")),
    ("The middle container takes red. The right one takes yellow. Blue goes left.", M("center", "right", "left")),
    ("Left: blue. Center: red. Right: yellow.", M("center", "right", "left")),
    ("left bin <- yellow, center bin <- blue, right bin <- red", M("right", "left", "center")),
    ("Red to the right bin, not the left one. Yellow to the left bin. Blue in the center.",
     M("right", "left", "center")),
    ("Put red in the left bin and blue in the right bin; the rest goes in the middle.", M("left", "center", "right")),
    ("Blue objects go right. Everything else goes in the center bin, except red, which goes left.",
     M("left", "center", "right")),
    ("Yellow in the left bin, blue in the center bin, and whatever is left in the right bin.",
     M("right", "left", "center")),
    ("Red goes left and yellow goes right.", M("left", "right", "center")),  # blue by elimination
    ("Put the crimson blocks in the right-hand box, golden ones in the left-hand box, navy ones in the middle.",
     M("right", "left", "center")),
    ("Put the yelow objects in the rigth bin, the red objects in the centre bin and the blu objects in the left bin.",
     M("center", "right", "left")),
    ("The bin to the left of the center one is for blue; the bin to the right of the middle is for red; yellow goes in the center.",
     M("right", "center", "left")),
    ("Red does not go in the left bin. Red goes in the center bin. Blue goes in the left bin.",
     M("center", "right", "left")),
    ("Place the blue objects in the left bin and the red objects in the right bin. Yellow ones go in the remaining bin.",
     M("right", "center", "left")),
    ("Into the right bin go the yellow objects; into the left bin go the blue ones; into the center bin go the red ones.",
     M("center", "right", "left")),
    ("Laita punaiset vasempaan laatikkoon, keltaiset keskelle ja siniset oikealle.", M("left", "center", "right")),
    ("Sort by colour: blue -> left, yellow -> right, red -> middle", M("center", "right", "left")),
    ("I want yellow on the left side, red on the right side and blue in between in the center.",
     M("right", "left", "center")),
    ("The left bin is for red objects, the center bin for yellow, the right bin for blue.",
     M("left", "center", "right")),
    ("Use the right bin for blue, the left bin for yellow and the center bin for red objects.",
     M("center", "left", "right")),
    ("All red objects go in the right bin. All blue objects go in the left bin. All yellow objects go in the middle bin.",
     M("right", "center", "left")),
    ("Every object goes into the center bin.", M("center", "center", "center")),
    ("Everything except blue goes in the left bin; blue goes right.", M("left", "left", "right")),
    ("Leave the yellow ones where they are. Red goes left, blue goes right.", M("left", None, "right")),
    ("From my perspective, red goes on my left, blue on my right, yellow in the middle.",
     M("right", "center", "left")),
    ("Kindly deposit blue items in the leftmost receptacle, yellow in the rightmost, and red in the one in the middle.",
     M("center", "right", "left")),
    ("Red belongs on the left. Yellow, center. Blue, right.", M("left", "center", "right")),
    ("Your task is to sort: blue on the right, yellow on the left, red centre.", M("center", "left", "right")),
    ("Group objects by colour. Left bin gets yellow, right bin gets red, middle bin gets blue.",
     M("right", "left", "center")),
    ("Yellow objects should end up in the right bin while blue objects end up in the left bin, and red objects in the middle.",
     M("center", "right", "left")),
    ("Take each red object and put it in the center bin. Then put each blue object in the right bin. Finally, the yellow ones go left.",
     M("center", "left", "right")),
    ("The blue ones go where the red ones would normally go, the red ones go in the middle, and yellow goes right.",
     M("center", "right", "left")),
    ("Put all the yellow and blue objects in the right bin and the red objects in the left bin.",
     M("left", "right", "right")),
    ("Swap the usual order: blue left, yellow center, red right.", M("right", "center", "left")),
    ("Only the red objects should be moved, into the right bin.", M("right", None, None)),
    ("Center bin: blue objects only. Left bin: red objects only. Right bin: yellow objects only.",
     M("left", "right", "center")),
    ("Put red objects in the bin on your right, and blue objects in the bin on your left; yellow in the center.",
     M("right", "center", "left")),
    ("Sort them so that the left box ends up with all the blue things and the right box with all the red things.",
     M("right", "center", "left")),
    # real object colours (organisers 2026-11-14): orange = red, purple = blue; the bins are orange boxes
    ("Put the orange objects in the left bin, the purple ones in the right bin and the yellow ones in the middle.",
     M("left", "center", "right")),
    ("Purple goes left, orange goes right, yellow center.", M("right", "center", "left")),
    ("Sort them: violet into the center box, orange into the left box, yellow into the right box.",
     M("left", "right", "center")),
    ("Put the purple objects in the left orange box, the orange objects in the right orange box and the yellow "
     "objects in the middle orange box.", M("right", "center", "left")),
    ("", M("left", "center", "right")),
    ("asdf qwerty", M("left", "center", "right")),
]


class TestTemplates(unittest.TestCase):
    def test_all_training_templates_all_permutations(self):
        bad = []
        for tpl in _templates():
            for cols in itertools.permutations(("red", "yellow", "blue")):
                for bins in itertools.permutations(("left", "center", "right")):
                    w = {f"c{i}": c for i, c in enumerate(cols)}
                    w.update({f"C{i}": c.capitalize() for i, c in enumerate(cols)})
                    w.update({f"b{i}": b for i, b in enumerate(bins)})
                    text = tpl.format(**w)
                    want = dict(zip(cols, bins))
                    got = parse_instruction(text).mapping
                    if got != want:
                        bad.append((text, want, got))
        self.assertEqual(bad, [], f"{len(bad)} template failures, first: {bad[:3]}")


class TestParaphrases(unittest.TestCase):
    def test_paraphrases(self):
        bad = []
        for text, want in PARAPHRASES:
            got = parse_instruction(text).mapping
            if got != want:
                bad.append((text, want, got))
        self.assertEqual(bad, [], "\n" + "\n".join(f"{t!r}\n   want {w}\n   got  {g}" for t, w, g in bad))

    def test_shape_rules(self):
        r = parse_instruction("Red cubes go in the left bin, red cylinders in the right bin, everything else in the center.")
        self.assertEqual(r.bin_for("red", "cube"), "left")
        self.assertEqual(r.bin_for("red", "cylinder"), "right")
        self.assertEqual(r.bin_for("blue", "cube"), "center")
        r = parse_instruction("Triangles go in the center bin. Otherwise red left, yellow center, blue right.")
        self.assertEqual(r.bin_for("blue", "triangle"), "center")
        self.assertEqual(r.bin_for("blue", "cube"), "right")


if __name__ == "__main__":
    unittest.main()
