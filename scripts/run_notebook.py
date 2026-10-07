from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / '.jupyter-runtime'
kernel = runtime / 'kernels' / 'project-local'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project Python', 'language': 'python',
}), encoding='utf-8')
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib-cache')
path = ROOT / 'notebooks/gdelt_energy_media_analysis.ipynb'
nb = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='project-local',
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
client = NotebookClient(nb, km=manager, timeout=300,
    resources={'metadata': {'path': str(ROOT)}}, allow_errors=False)
client.execute()
nbformat.validate(nb)
nbformat.write(nb, path)
print(f'Executed {sum(c.cell_type == "code" for c in nb.cells)} code cells successfully')
