"""Instruction -> sorting rules, offline and deterministic (no model, no network, < 1 ms).

    rules = parse_instruction("Sort the objects: red into right, blue into left, yellow into center.")
    rules.bin_for("red")                # -> "right"
    rules.bin_for("red", "cube")        # shape-aware when the instruction mentions shapes
    rules.mapping                       # {"red": "right", "blue": "left", "yellow": "center"}

How it works (robust to paraphrases rather than matching the 8 training templates):
  1. Normalise: lower-case, synonyms (centre/middle/central -> center, crimson -> red, block -> cube, ...), typo
     tolerance (difflib), a little Finnish/Swedish/German, "left of the center bin" -> left, "what's left" removed.
  2. Split into segments at sentence breaks / commas / semicolons; a segment without a bin is merged into the next
     one, a segment without an object into the previous one ("Put the red, yellow and blue objects in the left,
     center and right bins respectively" becomes one segment).
  3. In each segment, turn the token stream into mentions: ITEM (colour and/or shape, or ALL / REST / EXCEPT-set),
     BIN (position; a negated bin "not the left one" is dropped), IGNORE ("leave the red ones").
     Pairing: C C C B B B -> respectively; otherwise item-first segments assign the pending items to each bin as it
     appears, bin-first segments assign the following items to the current bin.
  4. Priorities: colour+shape > shape > colour > rest/except > all. "rest" fills only what nothing else claimed.
  5. Completion: colours still unassigned get the bins nobody uses (elimination; a permutation is the norm), never a
     forbidden pair ("red does not go left"). Nothing understood at all -> the fixed rung-1 mapping.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
# sort_fixed with the REAL colours (organisers, 2026-11-14): yellow -> left, purple -> center, orange -> right
FIXED_MAPPING = {"red": "right", "yellow": "left", "blue": "center"}

# --------------------------------------------------------------------------------------------- vocabulary
_COLOR_WORDS = {
    "red": "red", "reds": "red", "reddish": "red", "crimson": "red", "scarlet": "red", "maroon": "red",
    "ruby": "red", "cherry": "red",
    # The real objects are orange / yellow / purple (organisers, 2026-11-14): orange plays "red", purple "blue".
    "orange": "red", "oranges": "red", "orangey": "red", "tangerine": "red", "amber-red": "red",
    "yellow": "yellow", "yellows": "yellow", "yellowish": "yellow", "gold": "yellow", "golden": "yellow",
    "amber": "yellow", "lemon": "yellow",
    "blue": "blue", "blues": "blue", "bluish": "blue", "navy": "blue", "azure": "blue", "cyan": "blue",
    "cobalt": "blue", "sapphire": "blue", "teal": "blue",
    "purple": "blue", "purples": "blue", "purplish": "blue", "violet": "blue", "lilac": "blue", "lavender": "blue",
    "magenta": "blue", "indigo": "blue", "mauve": "blue", "blu": "blue", "bleu": "blue", "rd": "red",
    "yelow": "yellow", "yello": "yellow", "yelloww": "yellow",
    # Finnish / Swedish / German stems are handled in _foreign()
}
_SHAPE_WORDS = {
    # "block(s)" / "piece(s)" are generic object words here, not the cube shape
    "cube": "cube", "cubes": "cube", "cubic": "cube", "square": "cube",
    "squares": "cube", "dice": "cube", "die": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder", "cylindrical": "cylinder", "round": "cylinder",
    "circle": "cylinder", "circles": "cylinder", "circular": "cylinder", "disc": "cylinder", "discs": "cylinder",
    "disk": "cylinder", "disks": "cylinder", "can": "cylinder", "cans": "cylinder", "puck": "cylinder",
    "pucks": "cylinder", "tube": "cylinder", "tubes": "cylinder", "rod": "cylinder", "rods": "cylinder",
    "triangle": "triangle", "triangles": "triangle", "triangular": "triangle", "prism": "triangle",
    "prisms": "triangle", "wedge": "triangle", "wedges": "triangle", "pyramid": "triangle", "pyramids": "triangle",
    "tri": "triangle",
}
_BIN_WORDS = {
    "left": "left", "leftmost": "left", "lefthand": "left", "port": None,
    "right": "right", "rightmost": "right", "righthand": "right",
    "center": "center", "centre": "center", "middle": "center", "central": "center", "centered": "center",
    "centred": "center", "mid": "center", "midpoint": "center", "centermost": "center", "inner": None,
    "freebin": "free",   # "the remaining / other / empty bin": resolved to the unused bin at the end
}
_ALL_WORDS = {"all", "every", "everything", "each", "any", "anything"}
_REST_WORDS = {"rest", "remaining", "remainder", "others", "other", "else", "leftover", "leftovers"}
_EXCEPT_WORDS = {"except", "excluding", "besides", "but"}
_NEG_WORDS = {"not", "never", "no", "dont", "don't", "doesnt", "doesn't", "nor", "avoid", "neither", "without",
              "isnt", "isn't", "arent", "aren't", "shouldnt", "shouldn't", "mustnt", "mustn't"}
_IGNORE_VERBS = {"leave", "ignore", "skip", "keep"}
_FUZZY_VOCAB = sorted(set(_COLOR_WORDS) | set(_SHAPE_WORDS) | {k for k, v in _BIN_WORDS.items() if v})

# priorities (higher wins)
P_ALL, P_REST, P_COLOR, P_SHAPE, P_BOTH = 1, 2, 3, 4, 5


@dataclass
class Rules:
    """Result of parsing. `rules`: (colour|None, shape|None) -> bin or None (= leave it where it is)."""
    instruction: str
    rules: dict = field(default_factory=dict)
    prio: dict = field(default_factory=dict)
    forbidden: set = field(default_factory=set)       # (colour, bin) pairs the text excluded
    source: str = "parsed"                            # parsed | completed | fallback
    notes: list = field(default_factory=list)

    @property
    def mapping(self):
        """Colour -> bin (None = leave on the counter), what the planner needs when shapes don't matter."""
        out = {}
        for c in COLORS:
            b = self.bin_for(c)
            if b is None and (c, None) not in self.rules:  # only shape-qualified rules: use them if they agree
                bs = {v for (cc, s), v in self.rules.items() if cc == c and s is not None}
                b = bs.pop() if len(bs) == 1 else None
            out[c] = b
        return out

    @property
    def uses_shape(self):
        return any(s is not None for _, s in self.rules)

    def bin_for(self, color, shape=None):
        best, bp = None, -1
        for (c, s), b in self.rules.items():
            if c is not None and c != color:
                continue
            if s is not None and s != shape:
                continue
            p = self.prio.get((c, s), 0)
            if p > bp:
                best, bp = b, p
        return best

    def summary(self):
        return f"{self.source}: " + ", ".join(f"{c}->{self.bin_for(c)}" for c in COLORS) + (
            f" (+{sum(1 for _, s in self.rules if s)} shape rules)" if self.uses_shape else "")


# --------------------------------------------------------------------------------------------- normalise
def _foreign(w):
    """Tiny Finnish / Swedish / German lexicon by stem (Junction is in Helsinki)."""
    stems = (("punai", "red"), ("keltai", "yellow"), ("sini", "blue"), ("vasem", "left"), ("vasen", "left"),
             ("oikea", "right"), ("oikealle", "right"), ("keski", "center"), ("keskel", "center"),
             ("röd", "red"), ("gul", "yellow"), ("blå", "blue"), ("vänster", "left"), ("höger", "right"),
             ("mitt", "center"), ("rot", "red"), ("gelb", "yellow"), ("blau", "blue"), ("links", "left"),
             ("rechts", "right"), ("mitte", "center"))
    for s, v in stems:
        if w.startswith(s) and (len(w) <= len(s) + 6):
            return v
    return None


def _normalise(text):
    t = " " + text.lower().replace("’", "'").replace("-", " ") + " "
    # the bins themselves are orange boxes: "the left orange box" names a bin, not the orange objects
    t = re.sub(r"\borange\s+(bin|bins|box|boxes|container|containers|basket|baskets|tray|trays|crate|crates)\b", r"\1", t)
    t = re.sub(r"\bleft\s+hand(ed)?", "lefthand", t)
    t = re.sub(r"right\s+hand(ed)?", "righthand", t)
    t = re.sub(r"\bleft[\s]*most\b", "left", t)
    t = re.sub(r"\bright[\s]*most\b", "right", t)
    # "the bin to the left of the center (one/bin)" -> left
    t = re.sub(r"\b(left|right)\s+of\s+(the\s+)?(center|centre|middle|central)(\s+(bin|box|one|container))?",
               r"\1", t)
    t = re.sub(r"\bto the (left|right) of (the )?(center|centre|middle)\b", r"\1", t)
    # "left" that is not a position
    t = re.sub(r"\b(what'?s|whatever is|whatever's|anything|everything|that is|that's|is|are|ones?)\s+left\b(?!\s+(bin|box|one|side|container|basket|tray|crate))",
               r"\1 remaining", t)
    t = re.sub(r"\bleft\s+over\b", "remaining", t)
    # "where the red ones (would) normally go" -> the fixed rung-1 bin of that colour
    t = re.sub(r"\b(?:to |into |in )?where (?:the )?(red|yellow|blue)(?: ones| objects| items)? (?:would |should )?"
               r"(?:normally |usually |typically |by default )?(?:go|belong)\b",
               lambda m: f" in the {FIXED_MAPPING[m.group(1)]} bin ", t)
    t = re.sub(r"\b(remaining|other|last|free|empty|unused|spare|third|leftover|final)\s+(bin|box|container|one|basket|tray|crate)\b",
               " freebin ", t)
    t = re.sub(r"\ball right\b", "ok", t)
    t = re.sub(r"\bthe right (bin|box|container) for (each|every|the) (object|item|piece|one)s?\b", "its bin", t)
    t = re.sub(r"\bin the right place\b", "", t)
    t = re.sub(r"\b(rest|remainder) of (them|the objects|the items|the pieces)\b", r"\1", t)
    t = re.sub(r"\b(other than|apart from|aside from|with the exception of|save for)\b", " except ", t)
    t = re.sub(r"\b(n't|not)\b", " not ", t)
    t = re.sub(r"\b(\w+)n't\b", r"\1 not", t)
    return t


_SPLIT = re.compile(r"[.;!?\n]|,|\bthen\b|\bwhile\b|\bwhereas\b|\bmeanwhile\b|\bafterwards\b|\bfinally\b")


def _tokens(seg):
    out = []
    for w in re.findall(r"[a-zäöåü']+", seg):
        w = w.strip("'")
        if not w:
            continue
        if w in _COLOR_WORDS:
            out.append(("C", _COLOR_WORDS[w], w))
        elif w in _SHAPE_WORDS:
            out.append(("S", _SHAPE_WORDS[w], w))
        elif w in _BIN_WORDS:
            if _BIN_WORDS[w]:
                out.append(("B", _BIN_WORDS[w], w))
        elif w in _NEG_WORDS:
            out.append(("NEG", None, w))
        elif w in _EXCEPT_WORDS:
            out.append(("EXC", None, w))
        elif w in _REST_WORDS:
            out.append(("REST", None, w))
        elif w in _ALL_WORDS:
            out.append(("ALL", None, w))
        elif w in _IGNORE_VERBS:
            out.append(("IGN", None, w))
        elif (f := _foreign(w)) is not None:
            out.append(("C" if f in COLORS else "B", f, w))
        elif len(w) >= 4 and (m := difflib.get_close_matches(w, _FUZZY_VOCAB, n=1, cutoff=0.8)):
            v = m[0]
            out.append(("C", _COLOR_WORDS[v], w) if v in _COLOR_WORDS else
                       ("S", _SHAPE_WORDS[v], w) if v in _SHAPE_WORDS else ("B", _BIN_WORDS[v], w))
        else:
            out.append(("W", None, w))
    return out


# --------------------------------------------------------------------------------------------- mentions
def _mentions(toks):
    """Token stream -> list of ('I', item) / ('B', bin) / ('X', bin) negated bin / ('IGN',) markers.
    item = dict(kind=color|shape|both|all|rest|except, color, shape, colors(set), neg(bool))."""
    out = []
    i, n = 0, len(toks)
    neg_window = 0
    except_mode = False
    while i < n:
        k, v, w = toks[i]
        if k == "NEG":
            neg_window = 4
            i += 1
            continue
        if k == "EXC":
            # "but" only means except when a colour follows soon ("everything but red")
            if w != "but" or any(t[0] in "CS" for t in toks[i + 1:i + 3]):
                except_mode = True
            i += 1
            continue
        if k == "IGN":
            out.append(("IGN",))
            i += 1
            continue
        if k == "B":
            out.append(("X" if neg_window > 0 else "B", v))
            neg_window = 0
            i += 1
            continue
        if k in ("C", "S"):
            colors, shape = [], None
            j = i
            # collect "red, yellow and blue" style lists of colours, then an optional shape
            while j < n and toks[j][0] in ("C", "S", "W") and j - i < 8:
                if toks[j][0] == "C":
                    colors.append(toks[j][1])
                elif toks[j][0] == "S":
                    shape = toks[j][1]
                    j += 1
                    break
                elif toks[j][2] not in ("and", "or", "objects", "object", "ones", "one", "items", "item",
                                        "pieces", "piece", "coloured", "colored", "color", "colour", "the",
                                        "shaped", "shapes", "shape", "things", "thing", "a", "an", "&"):
                    break
                j += 1
            if except_mode:
                for c in colors or [None]:
                    out.append(("I", dict(kind="except", color=c, shape=shape, neg=False)))
                except_mode = False
            elif not colors:
                out.append(("I", dict(kind="shape", color=None, shape=shape, neg=neg_window > 0)))
            else:
                for c in colors:
                    out.append(("I", dict(kind="both" if shape else "color", color=c, shape=shape,
                                          neg=neg_window > 0)))
            i = max(j, i + 1)
            continue
        if k == "REST":
            out.append(("I", dict(kind="rest", color=None, shape=None, neg=False)))
        elif k == "ALL":
            # "all/every/each" only counts when no colour/shape follows right after ("all red objects")
            nxt = []
            for t in toks[i + 1:i + 3]:
                if t[0] in ("EXC", "NEG", "B"):
                    break
                nxt.append(t)
            if not any(t[0] in "CS" for t in nxt) and not (out and out[-1][0] == "I"
                                                                         and out[-1][1]["kind"] in ("rest", "all")):
                out.append(("I", dict(kind="all", color=None, shape=None, neg=False)))
        if neg_window > 0:
            neg_window -= 1
        i += 1
    # an EXCEPT list belongs to the preceding ALL/REST item: fold it in
    folded = []
    for m in out:
        if m[0] == "I" and m[1]["kind"] == "except":
            host = next((f for f in reversed(folded) if f[0] == "I" and f[1]["kind"] in ("all", "rest", "except_host")),
                        None)
            if host is None:  # "..., except red, which goes left": no "everything" to subtract from -> plain item
                c, s = m[1]["color"], m[1]["shape"]
                folded.append(("I", dict(kind="both" if c and s else "color" if c else "shape", color=c, shape=s,
                                         neg=False)))
                continue
            host[1]["kind"] = "except_host"
            host[1].setdefault("excl", []).append((m[1]["color"], m[1]["shape"]))
            continue
        folded.append(m)
    return folded


def _pair(ms):
    """Mentions of one segment -> list of (item, bin) and ignored items."""
    items = [m for m in ms if m[0] == "I"]
    bins = [m for m in ms if m[0] == "B"]
    pairs, ignored = [], []
    if not items:
        return pairs, ignored, [m[1] for m in ms if m[0] == "X"]
    negated_bins = [m[1] for m in ms if m[0] == "X"]
    if not bins:
        if any(m[0] == "IGN" for m in ms):
            ignored = [it[1] for it in items]
        return pairs, ignored, negated_bins
    seq = [m for m in ms if m[0] in ("I", "B")]
    kinds = "".join(m[0] for m in seq)
    # respectively: I^n B^n with n >= 2
    mres = re.fullmatch(r"(I{2,})(B{2,})", kinds)
    if mres and len(mres.group(1)) == len(mres.group(2)):
        n = len(mres.group(1))
        return [(seq[i][1], seq[n + i][1]) for i in range(n)], ignored, negated_bins
    if kinds[0] == "I":
        pending = []
        for m in seq:
            if m[0] == "I":
                pending.append(m[1])
            else:
                pairs += [(it, m[1]) for it in pending]
                pending = []
        if pending:  # trailing items with no bin after them: the last bin
            pairs += [(it, bins[-1][1]) for it in pending]
    else:
        cur = None
        for m in seq:
            if m[0] == "B":
                cur = m[1]
            elif cur is not None:
                pairs.append((m[1], cur))
    return pairs, ignored, negated_bins


# --------------------------------------------------------------------------------------------- main
def parse_instruction(text: str | None) -> Rules:
    r = Rules(instruction=text or "")
    if not text or not text.strip():
        r.rules = {(c, None): b for c, b in FIXED_MAPPING.items()}
        r.prio = {k: P_COLOR for k in r.rules}
        r.source = "fallback"
        r.notes.append("empty instruction")
        return r

    flip = bool(re.search(r"\b(my|viewer'?s?|audience'?s?|camera'?s?|observer'?s?) (left|right|perspective|point of view|view|side)\b"
                          r"|\bfrom (my|the viewer'?s?|the front|the audience'?s?) (perspective|point of view|view|side)\b"
                          r"|\bfacing the robot\b", text.lower()))
    norm = _normalise(text)
    segs = [s for s in _SPLIT.split(norm) if s.strip()]
    toks = [_tokens(s) for s in segs]
    # merge: segment with items but no bin -> into the next; bins but no items -> into the previous
    merged = []
    carry = []
    for tk in toks:
        tk = carry + tk
        has_b = any(t[0] == "B" for t in tk)
        has_i = any(t[0] in ("C", "S", "REST", "ALL") for t in tk)
        if has_i and not has_b and not any(t[0] == "IGN" for t in tk):
            carry = tk
            continue
        carry = []
        if has_b and not has_i and merged:
            merged[-1] = merged[-1] + tk
        else:
            merged.append(tk)
    if carry:
        if merged:
            merged[-1] = merged[-1] + carry
        else:
            merged.append(carry)

    claims = []          # (prio, color, shape, bin)
    rest_bins, all_bins, excl_rules = [], [], []
    for tk in merged:
        pairs, ignored, negated = _pair(_mentions(tk))
        for it, b in pairs:
            if flip and b in ("left", "right"):
                b = "right" if b == "left" else "left"
            if it.get("neg"):
                if it["color"]:
                    r.forbidden.add((it["color"], b))
                continue
            k = it["kind"]
            if k == "color":
                claims.append((P_COLOR, it["color"], None, b))
            elif k == "both":
                claims.append((P_BOTH, it["color"], it["shape"], b))
            elif k == "shape":
                claims.append((P_SHAPE, None, it["shape"], b))
            elif k == "rest":
                rest_bins.append(b)
            elif k == "all":
                all_bins.append(b)
            elif k == "except_host":
                excl_rules.append((b, it.get("excl", [])))
        for it in ignored:
            if it["kind"] in ("color", "both"):
                claims.append((P_BOTH if it["shape"] else P_COLOR, it["color"], it["shape"], None))
            elif it["kind"] == "shape":
                claims.append((P_SHAPE, None, it["shape"], None))
        # "red goes in the right bin, not the left one": the negated bin is a forbidden pair for those items
        for it, b in pairs:
            for nb in negated:
                if it.get("color"):
                    r.forbidden.add((it["color"], nb))

    for p, c, s, b in claims:  # later claims of equal priority override earlier ones ("actually, red goes left")
        key = (c, s)
        if r.prio.get(key, -1) <= p:
            r.rules[key], r.prio[key] = b, p
    colored = {c for (c, s) in r.rules if c is not None and s is None}
    for b, excl in excl_rules:  # "everything except red goes left"
        ex_colors = {c for c, _ in excl if c}
        for c in COLORS:
            if c not in ex_colors and (c, None) not in r.rules:
                r.rules[(c, None)], r.prio[(c, None)] = b, P_REST
        colored = {c for (c, s) in r.rules if c is not None and s is None}
    for b in rest_bins:
        for c in COLORS:
            if c not in colored:
                r.rules[(c, None)], r.prio[(c, None)] = b, P_REST
        colored = {c for (c, s) in r.rules if c is not None and s is None}
    for b in all_bins:
        r.rules.setdefault((None, None), b)
        r.prio.setdefault((None, None), P_ALL)

    if not r.rules:
        r.rules = {(c, None): b for c, b in FIXED_MAPPING.items()}
        r.prio = {k: P_COLOR for k in r.rules}
        r.source = "fallback"
        r.notes.append("nothing understood: fixed mapping")
        return r

    # "the remaining bin": whichever bin no explicit rule uses
    for key in [k for k, b in r.rules.items() if b == "free"]:
        used = {b for k, b in r.rules.items() if b not in (None, "free")}
        opts = [b for b in BINS if b not in used and (key[0], b) not in r.forbidden]
        if opts:
            r.rules[key] = opts[0]
            r.notes.append(f"{key} -> {opts[0]} (the remaining bin)")
        else:
            del r.rules[key]
    # completion by elimination: colours without any rule get an unused bin (never a forbidden pair)
    missing = [c for c in COLORS if r.bin_for(c) is None and (c, None) not in r.rules]
    if missing and re.search(r"\bonly\b", text.lower()):
        # "only the red objects should be moved": leave the others (0 points) instead of guessing (-1 if wrong)
        for c in missing:
            r.rules[(c, None)], r.prio[(c, None)] = None, P_REST
            r.notes.append(f"{c}: left alone ('only')")
        missing = []
    if missing and (None, None) not in r.rules and not r.uses_shape:
        used = {b for (c, s), b in r.rules.items() if b is not None}
        free = [b for b in BINS if b not in used]
        for c in missing:
            opts = [b for b in free if (c, b) not in r.forbidden]
            if len(missing) == 1 and len(opts) == 1 or (opts and len(free) == len(missing)):
                b = opts[0]
            elif FIXED_MAPPING[c] in opts:
                b = FIXED_MAPPING[c]
            elif opts:
                b = opts[0]
            else:
                continue
            r.rules[(c, None)], r.prio[(c, None)] = b, P_REST
            free.remove(b)
            r.source = "completed"
            r.notes.append(f"{c} -> {b} by elimination")
    # a forbidden pair that a rule still produces (contradiction): trust the explicit rule, but note it
    for (c, b) in r.forbidden:
        if r.rules.get((c, None)) == b and r.prio.get((c, None)) <= P_REST:
            del r.rules[(c, None)]
            r.notes.append(f"dropped {c}->{b} (forbidden)")
    return r


if __name__ == "__main__":
    import sys
    for line in (sys.argv[1:] or [l for l in sys.stdin.read().splitlines() if l.strip()]):
        print(f"{parse_instruction(line).summary():50s} | {line}")
