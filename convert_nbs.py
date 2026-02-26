import nbformat
import sys
import os

files = ["notebooks/01_arps_exploration.py", "notebooks/02_validation.py"]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    cells = content.split("# %%")
    nb = nbformat.v4.new_notebook()
    
    for cell in cells:
        cell = cell.strip()
        if not cell:
            continue
        nb.cells.append(nbformat.v4.new_code_cell(cell))
        
    out_file = file.replace(".py", ".ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Created {out_file}")
