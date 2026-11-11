"""Sequentially batch the audit into one CPU job and save each test result."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/user')

def main():
    cases = json.loads((ROOT / 'redteam/cases.json').read_text())
    summary = []
    for case in cases:
        start = time.monotonic()
        print('START', case, flush=True)
        process = subprocess.run(['bash', f'redteam/{case}/run.sh'], cwd=ROOT)
        summary.append({'case': case, 'exit_code': process.returncode,
                        'seconds': time.monotonic() - start})
        (ROOT / 'redteam/job_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('FINISH', case, process.returncode, flush=True)
    subprocess.run([sys.executable, 'redteam/analyze.py'], cwd=ROOT, check=True)
    if any(x['exit_code'] for x in summary):
        raise SystemExit('Some cases failed; inspect per-case evidence.')

if __name__ == '__main__':
    main()
