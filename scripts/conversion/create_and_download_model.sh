#!/bin/bash
# Run this script ON THE REMOTE INSTANCE (where you're SSH'd in)
# It creates a compressed archive of the model files

MODEL_DIR="$HOME/dakota_rl_training/outputs/ledger_test_400/weights/step_400"
OUTPUT_ARCHIVE="$HOME/model_step_400.tar.gz"

echo "Creating archive of model files..."
cd "$HOME/dakota_rl_training/outputs/ledger_test_400/weights"
tar -czf "$OUTPUT_ARCHIVE" step_400/

if [ $? -eq 0 ]; then
    echo " Archive created: $OUTPUT_ARCHIVE"
    echo ""
    echo "File size:"
    ls -lh "$OUTPUT_ARCHIVE"
    echo ""
    echo "Now download it from a NEW PowerShell window:"
    echo "  scp -i \$env:USERPROFILE\\.ssh\\prime_rl_key root@<INSTANCE_IP>:~/model_step_400.tar.gz archive/step0_legacy/root_artifacts/model_step_400.tar.gz"
    echo ""
    echo "Or check Prime Intellect dashboard for file download option"
else
    echo " Failed to create archive"
    exit 1
fi

