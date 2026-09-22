import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbformat as nbf
from nbclient import NotebookClient

from phase1_cells import get_phase1_cells
from phase2_cells import get_phase2_cells
from phase3_cells import get_phase3_cells
from phase4_cells import get_phase4_cells
from phase5_cells import get_phase5_cells
from phase6_cells import get_phase6_cells
from phase7_cells import get_phase7_cells
from final_cells import get_final_cells

nb = nbf.v4.new_notebook()

all_cells = (
    get_phase1_cells() +
    get_phase2_cells() +
    get_phase3_cells() +
    get_phase4_cells() +
    get_phase5_cells() +
    get_phase6_cells() +
    get_phase7_cells() +
    get_final_cells()
)

nb.cells = all_cells
notebook_path = "notebooks/churn_prediction.ipynb"

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated clean notebook with {len(nb.cells)} cells at {notebook_path}")
print("Executing notebook top to bottom...")

client = NotebookClient(nb, timeout=900, kernel_name="python3")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook successfully executed top to bottom and saved with all outputs!")
