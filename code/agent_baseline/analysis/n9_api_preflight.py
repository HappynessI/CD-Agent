"""Same-allocation workspace /models check; never print credentials or model list."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from .n8_missed_assets import write


def check(output):
    base = os.environ.get('DASHSCOPE_BASE_URL')
    key = os.environ.get('DASHSCOPE_API_KEY')
    if not base or not key: raise RuntimeError('Workspace CSV endpoint/key must be loaded before probing')
    proxy = any(bool(v) for k,v in os.environ.items() if 'proxy' in k.lower())
    row = dict(node=socket.gethostname(), endpoint_host=urllib.parse.urlparse(base).hostname,
               proxy_present=proxy, http_status=None, slurm_job=os.environ.get('SLURM_JOB_ID'))
    if proxy:
        write(output, row); raise RuntimeError('Compute proxy environment not cleared')
    request = urllib.request.Request(base.rstrip('/')+'/models', headers={'Authorization':'Bearer '+key})
    try:
        with urllib.request.urlopen(request, timeout=25) as response: row['http_status'] = response.status
    except urllib.error.HTTPError as exc:
        row['http_status'] = exc.code
    except OSError as exc:
        row['error_type'] = type(exc).__name__
    write(output, row); print(json.dumps(row), flush=True)
    if row['http_status'] != 200: raise RuntimeError('Workspace /models failed; original HTTP status preserved')
    return row


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER); a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Preflight must run in the API allocation')
    check(a.out)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if command: raise SystemExit(subprocess.call(command))

if __name__=='__main__': main()
