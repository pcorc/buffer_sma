"""The research notebook runs top to bottom against the current code (requires requirements-notebook.txt)."""
from pathlib import Path
import unittest

try:
    import nbformat
    from nbclient import NotebookClient
    HAS_NOTEBOOK = True
except ImportError:
    HAS_NOTEBOOK = False

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/ecr_backtest.ipynb'


@unittest.skipUnless(HAS_NOTEBOOK, 'Install requirements-notebook.txt to execute the notebook')
class NotebookTests(unittest.TestCase):
    def test_notebook_executes_without_errors(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        # Executes in memory; the tracked notebook keeps no outputs.
        NotebookClient(notebook, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': str(NOTEBOOK.parent)}}).execute()
