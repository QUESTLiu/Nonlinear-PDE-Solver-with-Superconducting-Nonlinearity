"""Mark text that is new or rephrased relative to the authors' original main.tex.

Usage (from the repository root), with ORIGINAL.tex the earlier draft, which this folder does not include:
    python paper/mark_changes.py paper/main.tex ORIGINAL.tex paper/main_marked.tex

Prose is compared sentence by sentence and displayed equations environment by
environment, after removing comments, whitespace and pure layout commands.
Anything that does not occur in the active text of the original is wrapped in
\\rev{...} (sentences, headings) or coloured with \\revcolor (equations, floats,
theorems). Setting \\renewcommand{\\revcolor}{} in the preamble removes every mark.
"""

import re
import sys

MATH_ENVS = ("align", "align*", "equation", "equation*", "gather", "gather*", "multline", "multline*")
ABBREV = {"Eq.", "Eqs.", "Ref.", "Refs.", "Sec.", "Secs.", "Fig.", "Figs.", "App.", "e.g.", "i.e.", "vs.", "cf.", "No."}


def strip_comments(src):
    out = []
    for line in src.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[: m.start()]
            if not line.strip():
                continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.S)
    return text


def unwrap(cmd, text):
    """Remove \\cmd{...} wrappers while keeping their content."""
    key = "\\" + cmd + "{"
    while True:
        i = text.find(key)
        if i < 0:
            return text
        depth, j = 1, i + len(key)
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        text = text[:i] + text[i + len(key): j - 1] + text[j:]


def norm_prose(t):
    t = t.replace("\\(", "$").replace("\\)", "$").replace("\\[", "$$").replace("\\]", "$$")
    t = t.replace("~", " ")
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    for tok in ("\\qquad", "\\quad", "\\left", "\\right", "\\,", "\\;", "\\!", "\\ "):
        t = t.replace(tok, "")
    t = re.sub(r"\s+", "", t)
    return t.replace("{}", "")


def norm_loose(t):
    """Wording-level key: drops math delimiters, display wrappers and light punctuation."""
    t = norm_prose(t)
    for tok in ("\\begin{align}", "\\end{align}", "\\begin{equation}", "\\end{equation}", "\\nonumber",
                "\\\\", "&", "$", ",", "."):
        t = t.replace(tok, "")
    return t


def norm_eq(t):
    t = norm_prose(t)
    t = t.replace("\\\\", "")
    for tok in ("\\begin{aligned}", "\\end{aligned}", "\\nonumber", "\\displaystyle", "\\qquad", "\\quad", "\\biggl", "\\biggr", "\\Bigl", "\\Bigr",
                "\\bigl", "\\bigr", "\\bigg", "\\Big", "\\big", "\\left", "\\right", "\\,", "\\;", "\\!", "&", "{}", "$"):
        t = t.replace(tok, "")
    t = t.replace("\\tfrac", "\\frac")
    return t.rstrip(".,")


def split_sentences(text):
    """Split a paragraph at sentence ends outside math and braces."""
    out, start, depth, math, i, n = [], 0, 0, None, 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("\\(", i) and math is None:
            math, i = ")", i + 2
            continue
        if text.startswith("\\)", i) and math == ")":
            math, i = None, i + 2
            if text[:i - 2].rstrip().endswith(".") and re.match(r"\s+[A-Z\\($\d]", text[i:]):
                out.append(text[start:i].strip())
                start = i
            continue
        if c == "\\":
            i += 2
            continue
        if c == "$":
            closing = math == "$"
            math = None if math == "$" else ("$" if math is None else math)
            i += 1
            if closing and text[:i - 1].rstrip().endswith((".", ".}")) and re.match(r"\s+[A-Z\\($\d]", text[i:]):
                out.append(text[start:i].strip())
                start = i
            continue
        if math is None:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c in ".?" and depth == 0:
                tail = text[i + 1:]
                words = text[start:i + 1].split()
                word = words[-1] if words else ""
                abbrev = word in ABBREV or (word == "al." and len(words) > 1 and words[-2] == "et")
                if re.match(r"\s+[A-Z\\($\d]", tail) and not abbrev:
                    out.append(text[start:i + 1].strip())
                    start = i + 1
        i += 1
    rest = text[start:].strip()
    if rest:
        out.append(rest)
    return out


class Marker:
    def __init__(self, original):
        orig = unwrap("cb", strip_comments(original))
        self.prose_corpus = norm_prose(orig)
        self.loose_corpus = norm_loose(orig)
        self.eq_corpus = norm_eq(orig)
        self.headings = {norm_prose(h).rstrip(".") for h in re.findall(r"\\(?:(?:sub)*section\*?|paragraph)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", orig)}
        self.new_sentences = self.old_sentences = self.new_eqs = self.old_eqs = 0

    def is_old_sentence(self, s):
        key = norm_prose(s).rstrip(".,")
        if key and key in self.prose_corpus:
            return True
        loose = norm_loose(s)
        return len(loose) > 20 and loose in self.loose_corpus

    def is_old_eq(self, body):
        key = norm_eq(body)
        return bool(key) and key in self.eq_corpus

    def prose(self, text):
        prefix = ""
        m = re.match(r"(\s*\\item\s+)", text)
        if m:
            prefix, text = m.group(1), text[m.end():]
        parts = []
        for s in split_sentences(text):
            if self.is_old_sentence(s):
                self.old_sentences += 1
                parts.append(s)
            else:
                self.new_sentences += 1
                parts.append("\\rev{" + s + "}")
        return prefix + " ".join(parts)

    def heading(self, line):
        """Return the (possibly marked) heading command and any text after it."""
        m = re.match(r"(\s*\\(?:sub)*section\*?\{|\s*\\paragraph\{)(.*)", line)
        head, rest = m.group(1), m.group(2)
        depth, j = 1, 0
        while depth:
            depth += {"{": 1, "}": -1}.get(rest[j], 0)
            j += 1
        title, after = rest[: j - 1], rest[j:]
        if norm_prose(title).rstrip(".") not in self.headings:
            title = "\\rev{" + title + "}"
        return head + title + "}", after.strip()


def collect(lines, i, end_pattern):
    block = [lines[i]]
    while not re.match(end_pattern, lines[i].strip()):
        i += 1
        block.append(lines[i])
    return block, i + 1


def main(clean_path, original_path, out_path):
    src = open(clean_path, encoding="utf-8").read()
    mk = Marker(open(original_path, encoding="utf-8").read())
    pre, body = src.split("\\begin{document}", 1)
    macros = ("\\newcommand{\\revcolor}{\\color{blue}}\n"
              "\\DeclareRobustCommand{\\rev}[1]{{\\revcolor #1}}\n"
              "\\pdfstringdefDisableCommands{\\def\\rev#1{#1}}\n")
    pre = pre.rstrip("\n") + "\n" + macros + "\n"
    lines = body.split("\n")
    out, para, i = [], [], 0

    def flush():
        if para:
            out.append(mk.prose(" ".join(l.strip() for l in para)))
            para.clear()

    passthrough = re.compile(r"\\(maketitle|appendix|clearpage|bibliography|bibliographystyle|label|title|author|affiliation|date|end\{document\}|begin\{itemize\}|end\{itemize\}|begin\{definition\}|end\{definition\})")
    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        if not s:
            flush()
            out.append("")
            i += 1
            continue
        m = re.match(r"\\begin\{([a-z*]+)\}", s)
        if m and m.group(1) in MATH_ENVS:
            flush()
            env = m.group(1).replace("*", r"\*")
            block, i = collect(lines, i, r"\\end\{" + env + r"\}")
            inner = "\n".join(block[1:-1])
            if mk.is_old_eq(inner):
                mk.old_eqs += 1
                out.extend(block)
            else:
                mk.new_eqs += 1
                out.append("\\begingroup\\revcolor")
                out.extend(block)
                out.append("\\endgroup")
            continue
        if s.startswith("\\[") and not s.endswith("\\]"):
            flush()
            block, i = collect(lines, i, r".*\\\]$")
            inner = "\n".join(block)[2:-2]
            if mk.is_old_eq(inner):
                out.extend(block)
            else:
                out.append("\\begingroup\\revcolor")
                out.extend(block)
                out.append("\\endgroup")
            continue
        if m and m.group(1) in ("figure", "figure*", "table", "table*"):
            flush()
            env = m.group(1).replace("*", r"\*")
            block, i = collect(lines, i, r"\\end\{" + env + r"\}")
            out.append(block[0])
            out.append("\\revcolor")
            out.extend(block[1:])
            continue
        if m and m.group(1) in ("theorem", "proposition"):
            flush()
            block, i = collect(lines, i, r"\\end\{" + m.group(1) + r"\}")
            out.append("\\begingroup\\revcolor")
            out.extend(block)
            out.append("\\endgroup")
            continue
        if m and m.group(1) == "center":
            flush()
            block, i = collect(lines, i, r"\\end\{center\}")
            inner = "\n".join(block[1:-1])
            if mk.is_old_eq(inner):
                out.extend(block)
            else:
                out.append("\\begingroup\\revcolor")
                out.extend(block)
                out.append("\\endgroup")
            continue
        if m and m.group(1) == "abstract":
            block, i = collect(lines, i, r"\\end\{abstract\}")
            out.append(block[0])
            out.append("\\rev{" + " ".join(b.strip() for b in block[1:-1]) + "}")
            out.append(block[-1])
            continue
        if re.match(r"\\((sub)*section\*?|paragraph)\{", s):
            flush()
            marked, after = mk.heading(line)
            out.append(marked)
            if after:
                para.append(after)
            i += 1
            continue
        if s.startswith("\\item"):
            flush()
            para.append(line)
            i += 1
            continue
        if passthrough.match(s):
            flush()
            out.append(line)
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    result = pre + "\\begin{document}" + "\n".join(out)
    open(out_path, "w", encoding="utf-8").write(result)
    print(f"sentences: {mk.new_sentences} marked, {mk.old_sentences} unchanged; displays: {mk.new_eqs} marked, {mk.old_eqs} unchanged")


if __name__ == "__main__":
    main(*sys.argv[1:4])
