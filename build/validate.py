import json, os, re, glob, sys
import nbformat
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = "https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/"
problems = 0
readme = open(f"{REPO}/README.md").read()
for path in sorted(glob.glob(f"{REPO}/notebooks/*.ipynb")):
    name = os.path.basename(path)
    nb = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(nb)
    except Exception as e:
        print(f"{name}: INVALID nbformat: {e}"); problems += 1
    raw = open(path).read()
    if REPO in raw:
        print(f"{name}: leaked local path"); problems += 1
    if GH + name not in nb.cells[0].source:
        print(f"{name}: first cell lacks its own Colab badge"); problems += 1
    if GH + name not in readme:
        print(f"{name}: not linked from README"); problems += 1
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    if errors:
        print(f"{name}: {len(errors)} error outputs"); problems += 1
    # outputs containing numpy scalar reprs or warnings
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code": continue
        for o in c.get("outputs", []):
            txt = "".join(o.get("text", "")) if o.output_type == "stream" else "".join(o.get("data", {}).get("text/plain", ""))
            if "np.float64(" in txt or "np.int64(" in txt:
                print(f"{name} cell {i}: numpy scalar repr in output"); problems += 1
            if "Warning" in txt:
                print(f"{name} cell {i}: warning in output: {txt.strip()[:100]}"); problems += 1
    # details blocks: each must have blank line after <summary> line and before </details>
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and "<details>" in c.source:
            s = c.source
            if not re.search(r"</summary>\n\n", s) or not re.search(r"\n\n</details>", s):
                print(f"{name} cell {i}: details block missing blank lines"); problems += 1
            if s.count("```") % 2:
                print(f"{name} cell {i}: unbalanced code fence"); problems += 1
    n_ex = sum(1 for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("### Exercise"))
    n_sol = sum(1 for c in nb.cells if c.cell_type == "markdown" and "<details>" in c.source)
    if n_ex != n_sol:
        print(f"{name}: {n_ex} exercises but {n_sol} solutions"); problems += 1
    size = os.path.getsize(path) / 1e6
    print(f"ok  {name}: {len(nb.cells)} cells, {n_ex} exercises, {size:.2f} MB")
print("PROBLEMS:", problems)
sys.exit(1 if problems else 0)
