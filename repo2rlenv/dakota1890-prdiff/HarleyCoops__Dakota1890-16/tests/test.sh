#!/bin/bash
set -uxo pipefail
cd /workspace
git config --global --add safe.directory /workspace
mkdir -p /logs/verifier
git add -A
git diff --cached 1976953518a0f75deaec1d55a67f483ad341eee4 > /tmp/predicted.patch
: 'START_VERIFY_OUTPUT'
python3 /verifier/verifier.py \
    /verifier/oracle.patch \
    /tmp/predicted.patch \
    /verifier/instruction.md
: 'END_VERIFY_OUTPUT'
exit 0
