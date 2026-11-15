#!/usr/bin/env python3
"""Exec a child command with DASHSCOPE_API_KEY loaded from a local key file.

Two formats are accepted: the console export (rows of `field,value` with an `apiKey` row and an
optional `openAiCompatible` endpoint row), or a file holding only the key. The key stays in memory
and is never printed. Usage: run_with_key.py KEY_FILE COMMAND [ARGS ...]
"""
import csv
import os
import sys
from pathlib import Path


def read_key(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    key = base_url = None
    for row in csv.reader(text.splitlines()):
        if len(row) < 2:
            continue
        field, value = row[0].strip().lower().replace('_', ''), row[1].strip()
        if field == 'apikey':
            key = value
        elif field == 'openaicompatible':
            base_url = value
    if key is None and len(text.split()) == 1:
        key = text.strip()
    if not key or not key.startswith('sk-'):
        raise SystemExit('Key file holds no valid key')
    return key, base_url


def main(argv):
    if len(argv) < 3:
        raise SystemExit('usage: run_with_key.py KEY_FILE COMMAND [ARGS ...]')
    key, base_url = read_key(argv[1])
    env = os.environ.copy()
    env['DASHSCOPE_API_KEY'] = key
    if base_url:
        env['DASHSCOPE_BASE_URL'] = base_url.rstrip('/')
    else:
        env.pop('DASHSCOPE_BASE_URL', None)
    os.execvpe(argv[2], argv[2:], env)


if __name__ == '__main__':
    main(sys.argv)
