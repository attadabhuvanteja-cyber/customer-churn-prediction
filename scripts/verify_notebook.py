import nbformat as nbf
from nbclient import NotebookClient

notebook_path = "notebooks/churn_prediction.ipynb"
print(f"Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

print(f"Total cells: {len(nb.cells)}")
print("Running complete clean kernel execution from top to bottom...")

client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

# Check each cell for any error outputs
error_count = 0
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for out in cell.get('outputs', []):
            if out.output_type == 'error':
                error_count += 1
                print(f"ERROR in Cell {idx}: {out.get('ename')}: {out.get('evalue')}")

if error_count == 0:
    print("SUCCESS: Notebook executed top to bottom with ZERO errors!")
else:
    print(f"FAILED: Found {error_count} errors in notebook cells!")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Updated notebook saved successfully.")
