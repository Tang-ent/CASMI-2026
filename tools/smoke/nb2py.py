"""Concatenate the code cells of a built notebook into one script: python nb2py.py <in.ipynb> <out.py>"""
import json
import sys

nb = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))
