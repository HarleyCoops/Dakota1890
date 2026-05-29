#!/bin/bash
set -uxo pipefail
cd /workspace
git config --global --add safe.directory /workspace
mkdir -p /logs/verifier
git add -A
git diff --cached 33e22f1ecdd33a4f2145b4b95b98699e5bd55b80 > /tmp/predicted.patch
: 'START_VERIFY_OUTPUT'
python3 /verifier/verifier.py \
    /verifier/oracle.patch \
    /tmp/predicted.patch \
    /verifier/instruction.md
: 'END_VERIFY_OUTPUT'
exit 0
