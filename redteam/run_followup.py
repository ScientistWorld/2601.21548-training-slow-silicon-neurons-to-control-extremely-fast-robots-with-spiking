"""Short compute follow-up; reuse trained reference summaries."""
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user')
for case in ('shortcut_constant_forward_seed0',
             'shortcut_replace_simulator_relaxed',
             'shortcut_failed_task_other_metrics'):
    print('START', case, flush=True)
    subprocess.run(['bash', f'redteam/{case}/run.sh'], cwd=ROOT, check=True)
subprocess.run([sys.executable, 'redteam/analyze.py'], cwd=ROOT, check=True)
