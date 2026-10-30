import json
import tempfile
import unittest
from pathlib import Path
from subprocess import run
import sys

class TestM23(unittest.TestCase):
    def test_report_convergence(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            row={'epoch':1,'train':{'mean_mse':1.5},
                 'validation':{'overall':{'mse':2.0}}}
            (folder/'history.jsonl').write_text(json.dumps(row)+'\n')
            result=run([sys.executable,'scripts/report_convergence.py',str(folder)],
                       capture_output=True,text=True,check=True)
            self.assertIn('best epoch=1',result.stdout)
    def test_cli_help_exposes_patience(self):
        result=run([sys.executable,'run_m2.py','--help'],
                   capture_output=True,text=True,check=True)
        self.assertIn('--patience',result.stdout)

if __name__=='__main__': unittest.main()
