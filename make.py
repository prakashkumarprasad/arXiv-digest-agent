#!/usr/bin/env python3
"""Minimal make wrapper for Windows."""
import sys, os, subprocess, re, shlex

def parse_makefile(path):
    targets, cur, cmds = {}, None, []
    for line in open(path):
        s = line.rstrip()
        if not s or s.startswith('#'): continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*)\s*:', s)
        if m and not s.startswith('\t'):
            if cur: targets[cur] = cmds
            cur, cmds = m.group(1), []
        elif s.startswith('\t') and cur:
            cmds.append(s.lstrip('\t'))
    if cur: targets[cur] = cmds
    return targets

def run(cmd):
    if cmd.startswith('python '):
        cmd = cmd.replace('python ', f'"{sys.executable}" ', 1)
    parts = shlex.split(cmd)
    r = subprocess.run(parts, shell=False)
    if r.returncode: sys.exit(r.returncode)

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h','--help'):
        print("Usage: make <target>"); sys.exit(0)
    target, mk = sys.argv[1], 'Makefile'
    if len(sys.argv) > 2 and sys.argv[2] == '-f':
        mk, target = sys.argv[3], sys.argv[1] if len(sys.argv) > 3 else target
    if not os.path.exists(mk): print(f"make: *** No rule to make target '{target}'."); sys.exit(1)
    targets = parse_makefile(mk)
    if target not in targets: print(f"make: *** No rule to make target '{target}'."); sys.exit(1)
    for cmd in targets[target]: run(cmd)

if __name__ == '__main__': main()
