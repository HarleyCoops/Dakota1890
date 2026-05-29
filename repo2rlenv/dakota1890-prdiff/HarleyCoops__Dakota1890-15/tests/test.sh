#!/bin/bash
set -uxo pipefail
cd /workspace
git config --global --add safe.directory /workspace
mkdir -p /logs/verifier
git add -A
git diff --cached 3ee7c470253f0394330dac38585f052b5b257981 > /tmp/predicted.patch
: 'START_VERIFY_OUTPUT'
python3 /verifier/verifier.py \
    /verifier/oracle.patch \
    /tmp/predicted.patch \
    /verifier/instruction.md
: 'END_VERIFY_OUTPUT'
exit 0
