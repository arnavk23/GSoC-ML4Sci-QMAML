"""Compile a temporary copy of sn-article.tex using the article stand-in (sn-jnl.cls is not installed locally)."""
import os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, "sn-article.tex"), encoding="utf-8").read()
old = r"\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}"
new = "\\documentclass[11pt]{article}\n\\usepackage[margin=1.1in]{geometry}\n\\usepackage{url}\n\\input{sn-shim}"
assert old in s
open(os.path.join(HERE, "_check_inlined.tex"), "w", encoding="utf-8").write(s.replace(old, new, 1))
ok = True
for i in (1, 2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "_check_inlined.tex"],
                       cwd=HERE, capture_output=True, text=True)
    ok = ok and r.returncode == 0
    print(f"pass {i}: exit {r.returncode}")
log = open(os.path.join(HERE, "_check_inlined.log"), encoding="utf-8", errors="ignore").read()
print("errors:", [l for l in log.splitlines() if l.startswith("!")][:3])
print("undefined refs/cites:", log.count("undefined"))
print([l for l in log.splitlines() if "Output written" in l])
for ext in ("tex", "aux", "log", "pdf", "out"):
    p = os.path.join(HERE, f"_check_inlined.{ext}")
    if os.path.exists(p): os.remove(p)
