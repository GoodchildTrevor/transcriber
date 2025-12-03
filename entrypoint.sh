#!/bin/bash

echo "🔧 Starting entrypoint..."

# --- 1. Ensure cache dirs exist
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
mkdir -p "$HF_HOME" 2>/dev/null || true

# --- 2. cuDNN: enforce use of pip-installed version (critical for cu128)
CUDNN_LIB_DIR="/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -d "$CUDNN_LIB_DIR" ]; then
    echo "✅ Found cuDNN in venv: $CUDNN_LIB_DIR"
    export LD_LIBRARY_PATH="$CUDNN_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "🔧 LD_LIBRARY_PATH updated to prefer pip cuDNN"
else
    echo "⚠️ WARNING: cuDNN not found in venv (expected at $CUDNN_LIB_DIR)"
    echo "   → Ensure 'nvidia-cudnn-cu12' is installed via pip"
fi

# --- 3. Hugging Face auth
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ WARNING: HF_TOKEN not set — gated models will likely fail"
else
    echo "🔑 HF_TOKEN is set (length: ${#HF_TOKEN})"
    if /opt/venv/bin/python -c "
import os, sys
token = os.getenv('HF_TOKEN')
if not token:
    sys.exit(1)
try:
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print('✅ HF login via Python succeeded')
except Exception as e:
    print(f'❌ HF login failed: {e}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "✅ Hugging Face login completed"
else
    echo "❌ HF login failed — continuing (token may be passed directly to model)"
fi
fi

--- 4. Optional: diagnostics (uncomment for debug)
echo "📊 Diagnostics:"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
if command -v /opt/venv/bin/python >/dev/null; then
    /opt/venv/bin/python -c "
import torch, sys
print(f'   torch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version (runtime): {torch.version.cuda}')
    print(f'   cuDNN version: {torch.backends.cudnn.version()}')
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    try:
        from nvidia import cudnn
        print(f'   nvidia-cudnn-cu12 version: {cudnn.__version__}')
    except Exception as e:
        print(f'   nvidia-cudnn-cu12: not importable ({e})')
" 2>/dev/null || echo "   Python diagnostics failed"
fi

echo "🚀 Starting application: $*"
exec "$@"
