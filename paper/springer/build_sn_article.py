"""Assemble the single-file Springer Nature manuscript (sn-article.tex) and an Overleaf upload package.

Regenerates figures and tables from the saved results, inlines every \\input, wraps the body in the
sn-jnl document class header, and zips sn-article.tex + figures/ for upload into a project created from
the Springer Nature LaTeX template on Overleaf (which supplies sn-jnl.cls and the .bst files).
"""
import os, re, zipfile, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
subprocess.run([PY, os.path.join(HERE, "make_figures.py")], check=True, cwd=HERE)
subprocess.run([PY, os.path.join(HERE, "make_figures_extra.py")], check=True, cwd=HERE)
subprocess.run([PY, os.path.join(HERE, "make_tables.py")], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
subprocess.run([PY, os.path.join(HERE, "make_supp_tables.py")], check=True, cwd=HERE, stdout=subprocess.DEVNULL)

HEADER = r"""%% Springer Nature LaTeX template (sn-jnl). Manuscript source; figures in ./figures (vector PDF).
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathrsfs}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%
\usepackage{algorithm}%
\usepackage{algorithmicx}%
\usepackage{algpseudocode}%
\usepackage{listings}%
\usepackage{tikz}
\usepackage[section]{placeins}
\usetikzlibrary{arrows.meta}
\definecolor{qorange}{HTML}{D55E00}
\definecolor{qblue}{HTML}{0072B2}
\raggedbottom
\begin{document}

"""
FOOTER = "\n\\end{document}\n"

body = open(os.path.join(HERE, "body.tex"), encoding="utf-8").read()


def inline(m):
    name = m.group(1)
    if not name.endswith(".tex"): name += ".tex"
    return open(os.path.join(HERE, name), encoding="utf-8").read().rstrip() + "\n"


body = re.sub(r"\\input\{([^}]+)\}", inline, body)
out = HEADER + body + FOOTER
open(os.path.join(HERE, "sn-article.tex"), "w", encoding="utf-8").write(out)

zpath = os.path.join(HERE, "overleaf_package.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(HERE, "sn-article.tex"), "sn-article.tex")
    fdir = os.path.join(HERE, "figures")
    for f in sorted(os.listdir(fdir)):
        if f.endswith(".pdf"): z.write(os.path.join(fdir, f), f"figures/{f}")
    z.write(os.path.join(HERE, "supplementary.tex"), "supplementary.tex")
    for f in sorted(os.listdir(HERE)):
        if f.startswith("supp_") and f.endswith(".tex"): z.write(os.path.join(HERE, f), f)
print("wrote sn-article.tex (%d lines) and %s" % (out.count("\n"), os.path.basename(zpath)))
