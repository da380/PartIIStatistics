"""Rebuild every notebook from its source file, then validate the results.

    python build/build_all.py           # all notebooks
    python build/build_all.py 3 7       # just notebooks 3 and 7
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
nums = [int(a) for a in sys.argv[1:]] or range(1, 9)
for n in nums:
    print(f"=== notebook {n}")
    subprocess.run([sys.executable, os.path.join(HERE, f"nb{n:02d}.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "validate.py")], check=True)
