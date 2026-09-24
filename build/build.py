"""Tiny notebook builder: cells are declared in Python, executed with nbclient, and
written into the repo's notebooks/ directory with outputs rendered.

Data files are referenced by their raw-GitHub URL in the student-facing source, but
are read from the local data/ folder during execution so the build works before push.
Solution code hidden inside <details> blocks is also executed (in a throwaway copy)
so broken solutions fail the build.
"""
import re, sys, os, copy
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks")
DATA_URL = "https://raw.githubusercontent.com/da380/PartIIStatistics/main/data/"
DATA_LOCAL = os.path.join(REPO, "data") + "/"
GH = "https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/"


def md(src):
    return ("markdown", src.strip("\n"))


def code(src):
    return ("code", src.strip("\n"))


def badge(fname):
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GH}{fname})"


def header(fname, title, goals, tools):
    """Standard opening cell: badge, title, learning goals, tools used."""
    g = "\n".join(f"- {x}" for x in goals)
    t = "\n".join(f"- {x}" for x in tools)
    return md(f"""
{badge(fname)}

# {title}

**What you should get out of this notebook**

{g}

**Tools introduced**

{t}
""")


SETUP = code('''
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

rng = np.random.default_rng(1)          # random number generator with a fixed seed
plt.rcParams["figure.figsize"] = (8, 4.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
DATA = "https://raw.githubusercontent.com/da380/PartIIStatistics/main/data/"
''')


def exercise(n, text):
    return md(f"### Exercise {n}\n\n{text.strip()}")


def solution(text):
    return md(f"""
<details>
<summary><b>Solution</b> (click to expand)</summary>

{text.strip()}

</details>
""")


def scratch():
    return code("# Your code here\n")


def _strip(nb):
    for c in nb.cells:
        c.metadata = {}
        if c.cell_type == "code":
            c.execution_count = None
            for o in c.get("outputs", []):
                if "execution_count" in o:
                    o["execution_count"] = None
                if o.get("output_type") in ("display_data", "execute_result"):
                    o["metadata"] = {}
    return nb


def _make(cells):
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "markdown" else new_code_cell(src))
    return nb


def _execute(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace(DATA_URL, DATA_LOCAL)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": OUT}})
    client.execute()
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace(DATA_LOCAL, DATA_URL)
    return nb


FENCE = re.compile(r"```python\n(.*?)```", re.S)


def build(fname, cells, check_solutions=True):
    os.makedirs(OUT, exist_ok=True)
    nb = _execute(_make(cells))
    _strip(nb)
    nbformat.write(nb, os.path.join(OUT, fname))
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    n_code = len(nb.cells) - n_md
    print(f"built {fname}: {n_md} markdown + {n_code} code cells")

    if check_solutions:
        # Rebuild: all code cells in order, and the code inside each solution block
        # inserted at the point where the solution appears.
        chk, is_sol = [], []
        for kind, src in cells:
            if kind == "code":
                chk.append(("code", src)); is_sol.append(False)
            elif "<details>" in src:
                for block in FENCE.findall(src):
                    chk.append(("code", block)); is_sol.append(True)
        nbc = _make(chk)
        _execute(nbc)
        for c, flag in zip(nbc.cells, is_sol):       # show what the solution code printed
            if not flag: continue
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    print("   [solution] " + "".join(o["text"]).rstrip().replace("\n", "\n   [solution] "))
        print(f"solutions verified: {sum(1 for k, s in cells if k == 'markdown' and '<details>' in s)} solution blocks ran cleanly")
