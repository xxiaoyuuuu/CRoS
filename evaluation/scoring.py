"""Rule-based answer scoring with two labels: answer_correct (default) judges the <answer>; from_cot also accepts the gold answer stated as the conclusion of the CoT."""

import ast
import functools
import json
import re
from fractions import Fraction

import reasoning_gym
from reasoning_gym.utils import extract_answer


@functools.lru_cache(maxsize=None)
def _dataset(name):
    return reasoning_gym.create_dataset(name, size=100, seed=42)  # only used for its scorer


def rg_strict(name, response, entry):
    """The Reasoning Gym scorer on the <answer> (score 1 = correct)."""
    answer = extract_answer(response, tag_name="answer")
    try:
        return float(_dataset(name).score_answer(answer=answer, entry=entry)) >= 1.0
    except Exception:
        return False


def answer_correct(name, response, entry):
    """rg_strict, or the gold answer in another format. puzzle24 / countdown: the expression must equal the target
    exactly and use exactly the given numbers (the Reasoning Gym scorer does not check the numbers)."""
    answer, gold = extract_answer(response, tag_name="answer"), entry.get("answer")
    if answer is None:
        return False
    if name in ("puzzle24", "countdown"):
        return expression_correct(name, answer, entry)
    if rg_strict(name, response, entry):
        return True
    return gold is not None and format_ok_answer(name, answer, str(gold), entry) is not None


def from_cot(name, response, entry):
    """rg_strict, the gold answer in another format, or the gold answer as the stated conclusion of the CoT."""
    if rg_strict(name, response, entry):
        return True
    answer, gold = extract_answer(response, tag_name="answer"), entry.get("answer")
    if answer is None or gold is None:
        return False
    gold = str(gold)
    return bool(format_ok_answer(name, answer, gold, entry) or cot_rescue(name, response, gold, entry))


def score_all(name, response, entry):
    return {"answer_correct": answer_correct(name, response, entry), "from_cot": from_cot(name, response, entry)}


# ---------- puzzle24 / countdown ----------

def exact_eval(expr):
    def go(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -go(n.operand) if isinstance(n.op, ast.USub) else go(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = go(n.left), go(n.right)
            if isinstance(n.op, ast.Div):
                return a / b
            return a + b if isinstance(n.op, ast.Add) else a - b if isinstance(n.op, ast.Sub) else a * b
        raise ValueError("unsupported expression")
    return go(ast.parse(expr, mode="eval").body)


def expression_correct(name, answer, entry):
    s = unwrap(ws(answer)).replace("×", "*").replace("÷", "/").replace("−", "-")
    if not s or "=" in s or "\n" in s:
        return False
    try:
        value = exact_eval(s)
    except Exception:
        return False
    meta = entry.get("metadata", {})
    target = 24 if name == "puzzle24" else meta.get("target")
    used = sorted(int(x) for x in re.findall(r"\d+", s))
    return target is not None and value == target and used == sorted(meta.get("numbers", []))


# ---------- format tolerance: same answer, different formatting ----------

NO_WS_CASE = {"string_manipulation"}  # the task itself is about spaces / case
NO_NUMERIC = {"fraction_simplification", "base_conversion", "binary_alternation", "count_bits", "bitwise_arithmetic",
              "binary_matrix", "spell_backward", "palindrome_generation", "string_insertion", "string_manipulation",
              "string_splitting", "string_synthesis", "letter_jumble", "word_ladder", "ab", "bf", "caesar_cipher",
              "cryptarithm", "number_format", "emoji_mystery"}
ALGEBRA_VAR = {"simple_equations", "polynomial_equations", "polynomial_multiplication", "complex_arithmetic",
               "simple_integration", "intermediate_integration", "basic_arithmetic", "decimal_arithmetic",
               "decimal_chain_sum", "chain_sum", "gcd", "lcm", "prime_factorization", "power_function", "products",
               "time_intervals", "leg_counting", "count_primes"}
SYMPY_DS = {"simple_integration", "intermediate_integration", "polynomial_multiplication", "polynomial_equations",
            "complex_arithmetic"}
NO_LIST = {"string_manipulation", "sentence_reordering", "word_sorting"}


def ws(s):  # per line: trim, collapse spaces; drop empty lines
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in s.replace("\r", "").split("\n")]
    return "\n".join(l for l in lines if l)


def unwrap(s):  # drop $..$, \boxed{}, \text{}, quotes, a final period, an "answer:" prefix
    s = s.strip()
    s = re.sub(r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:：=]?\s*", "", s, flags=re.I)
    for _ in range(2):
        s = s.strip()
        m = re.fullmatch(r"\$+(.*?)\$+", s, flags=re.S); s = m.group(1).strip() if m else s
        m = re.fullmatch(r"\\boxed\{(.*)\}", s, flags=re.S); s = m.group(1).strip() if m else s
        m = re.fullmatch(r"\\\[(.*)\\\]", s, flags=re.S); s = m.group(1).strip() if m else s
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        if len(s) >= 2 and s[0] in "\"'`“‘" and s[-1] in "\"'`”’": s = s[1:-1]
        s = re.sub(r"[.。]\s*$", "", s).strip()
    return s


def strip_var(s):  # "x = 5" -> "5"
    return re.sub(r"^\s*[A-Za-z]\s*=\s*", "", s)


def tonum(s):
    s = s.strip().replace("$", "").replace(",", "").replace(" ", "").replace("\\%", "%")
    s = re.sub(r"(°|degrees?|cm|m|kg|units?)$", "", s, flags=re.I)
    if re.match(r"^0\d", s.lstrip("-")): return None  # leading zero: not a number
    m = re.fullmatch(r"(-?)\\d?frac\{(-?\d+)\}\{(\d+)\}", s)
    if m: return (-1 if m.group(1) else 1) * Fraction(int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m and int(m.group(2)): return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", s)
    if m: return Fraction(m.group(1)) / 100
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", s):
        try: return Fraction(s)
        except Exception: return None
    return None


def item_norm(s, ds):
    s = unwrap(s)
    if ds in ALGEBRA_VAR: s = strip_var(s)
    n = tonum(s) if ds not in NO_NUMERIC else None
    return ("#", n) if n is not None else ("s", s.casefold() if ds not in NO_WS_CASE else s)


def split_items(s):
    return [x for x in re.split(r"[,\n;]+|\s+|[\[\]\(\)\{\}]", s) if x.strip()]


def rows_of(s):  # grid: tokens per line, skipping lines without letters or digits
    out = []
    for l in s.replace("\r", "").split("\n"):
        if not re.search(r"[0-9A-Za-z_✅❌]", l): continue
        toks = [t for t in re.split(r"[,\s]+|[\[\]\(\)\{\}]", l) if t]
        if toks: out.append(toks)
    return out


def try_json(s):
    s = unwrap(s)
    try: return json.loads(s)
    except Exception: pass
    try: return ast.literal_eval(s)
    except Exception: return None


def words_np(s):
    return [w for w in re.sub(r"[^\w\s]", " ", s).casefold().split() if w]


def sympy_equiv(ds, a, g, entry):
    import sympy
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                            standard_transformations)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)

    def prep(s, var):
        s = unwrap(s); s = strip_var(s)
        s = s.replace("^", "**").replace("ln(", "log(").replace("\\", "")
        s = re.sub(r"\*\s*\+\s*C\s*$", " + C", s); s = re.sub(r"\*\s*C\s*$", " + C", s)
        if var and var != "X": s = re.sub(r"\bX\b", var, s)
        return s
    md = entry.get("metadata", {}) if entry else {}
    try:
        if ds in ("simple_integration", "intermediate_integration"):
            var = md.get("variable", "x"); x = sympy.Symbol(var); C = sympy.Symbol("C")
            ld = {var: x, "C": C, "e": sympy.E, "pi": sympy.pi}
            ua = parse_expr(prep(a, var), local_dict=ld, transformations=tr)
            integrand = md.get("integrand")
            if integrand is None:
                ug = parse_expr(prep(g, var), local_dict=ld, transformations=tr)
                return sympy.simplify(sympy.diff(ua - ug, x)) == 0 and (ua - ug).free_symbols <= {C}
            ig = parse_expr(str(integrand), local_dict=ld, transformations=tr)
            return sympy.simplify(sympy.diff(ua, x) - ig) == 0
        ld = {"i": sympy.I, "I": sympy.I, "e": sympy.E, "pi": sympy.pi}
        ua = parse_expr(prep(a, None), local_dict=ld, transformations=tr, evaluate=False)
        ug = parse_expr(prep(g, None), local_dict=ld, transformations=tr)
        if ds in ("polynomial_multiplication", "polynomial_equations"):
            if ua.has(sympy.Mul) and any(isinstance(m, sympy.Mul) and any(
                    isinstance(f, (sympy.Add, sympy.Pow)) and (isinstance(f, sympy.Add) or isinstance(f.base, sympy.Add))
                    for f in m.args) for m in sympy.preorder_traversal(ua)):
                return False  # the answer must already be expanded
        return sympy.simplify(sympy.expand(ua - ug)) == 0
    except Exception:
        return False


def format_ok_answer(ds, a, g, entry):
    """Name of the formatting rule under which answer a equals gold g, or None."""
    if a is None or g is None: return None
    A, G = ws(a), ws(g)
    if ds not in NO_WS_CASE:
        if A == G: return "ws"
        if A.casefold() == G.casefold(): return "case"
    ua, ug = unwrap(A), unwrap(G)
    cf = (lambda s: s) if ds in NO_WS_CASE else (lambda s: s.casefold())
    if cf(ua) == cf(ug): return "wrap"
    if ds in ALGEBRA_VAR and cf(strip_var(ua)) == cf(strip_var(ug)): return "var_eq"
    if ds == "fraction_simplification":  # only the notation may differ; the fraction must be fully reduced
        md = entry.get("metadata", {}) if entry else {}
        m = re.fullmatch(r"\\?d?frac\{(-?\d+)\}\{(\d+)\}|(-?\d+)\s*/\s*(\d+)",
                         ua.replace("$", "").replace(" ", "").replace("\\d", "\\"))
        if m:
            n_, d_ = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(4)))
            if n_ == md.get("simplified_numerator") and d_ == md.get("simplified_denominator"): return "frac_form"
        return None
    if ds not in NO_NUMERIC:
        na, ng = tonum(strip_var(ua) if ds in ALGEBRA_VAR else ua), tonum(ug)
        if na is not None and ng is not None and na == ng: return "numeric"
    if ds == "sentence_reordering":
        return "sentence" if words_np(ua) == words_np(ug) and words_np(ug) else None
    if ug.startswith("{") or ug.startswith("["):
        ja, jg = try_json(ua), try_json(ug)
        if ja is not None and jg is not None and ja == jg: return "json"
        if jg is not None: return None  # a structured gold answer is compared as a structure only
    if ds not in NO_LIST:
        ra, rg_ = rows_of(A), rows_of(G)
        if len(rg_) >= 2:  # multi-line gold (grid): same tokens row by row
            if len(ra) == len(rg_) and all(len(x) == len(y) and all(item_norm(p, ds) == item_norm(q, ds) for p, q in zip(x, y))
                                           for x, y in zip(ra, rg_)): return "grid"
        else:
            ia, ig = split_items(ua), split_items(ug)
            if len(ig) >= 2 and len(ia) == len(ig) and all(item_norm(p, ds) == item_norm(q, ds) for p, q in zip(ia, ig)):
                return "list"
    if not re.search(r"\s", ug) and ds not in NO_WS_CASE and re.sub(r"\s+", "", cf(ua)) == cf(ug): return "nospace"
    if ds in SYMPY_DS and sympy_equiv(ds, ua, ug, entry): return "sympy"
    return None


# ---------- CoT rescue: the reasoning concludes with the gold answer ----------

MARK = re.compile(r"(final answer|the answer is|answer\s*[:=：]|therefore[,:]?|thus[,:]?|so,? the (?:final )?answer|"
                  r"result is|hence[,:]?|conclusion[:：]?|final result[:：]?|output is|output[:：])", re.I)


def conclusions(cot, gold):
    """Candidate final conclusions of the CoT: last \\boxed{}, lines after a conclusion marker, last lines, last block."""
    c = []
    ms = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", cot)
    if ms: c.append(ms[-1])
    ne = [(i, l) for i, l in enumerate(l.rstrip() for l in cot.split("\n")) if l.strip()]
    hits = []
    for k, (i, l) in enumerate(ne):
        m = MARK.search(l)
        if not m: continue
        rest = l[m.end():].strip(" :：=*_-\t")
        if rest and len(rest) <= 400: hits.append(rest)
        elif k + 1 < len(ne) and len(ne[k + 1][1]) <= 400: hits.append(ne[k + 1][1].strip())
    c += hits[-3:]
    c += [l.strip() for _, l in ne[-3:] if len(l) <= 400]
    ng = gold.count("\n") + 1
    if ng >= 2 and len(ne) >= ng:
        c.append("\n".join(l for _, l in ne[-ng:]))
        if len(ne) >= ng + 1: c.append("\n".join(l for _, l in ne[-ng - 1:-1]))
    return c


def cot_rescue(ds, response, gold, entry):
    i = response.rfind("<answer>")
    cot = response[:i] if i >= 0 else response
    for cand in conclusions(cot, gold):
        cand = re.sub(r"^\*+|\*+$", "", cand.strip()).strip()
        if cand and (ws(cand) == ws(gold) or format_ok_answer(ds, cand, gold, entry) is not None):
            return True
    return False
