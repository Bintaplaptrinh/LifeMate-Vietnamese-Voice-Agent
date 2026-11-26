"""Serve the LifeMate v4 QwenTune checkpoint with SGLang."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

MODEL_ALIAS = "lifemate-qwentune-hf"


def server_command(model, port):
    model = Path(model).resolve()
    template = model / "chat_template.jinja"
    if not template.is_file():
        raise FileNotFoundError(f"Missing QwenTune chat template: {template}")
    return [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(model), "--served-model-name", MODEL_ALIAS,
        "--chat-template", str(template),
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "float16",
        "--random-seed", "42",
        "--attention-backend", "triton", "--linear-attn-backend", "triton",
        "--mamba-ssm-dtype", "float16", "--disable-cuda-graph",
        "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--context-length", "16384", "--mem-fraction-static", "0.90",
        "--max-mamba-cache-size", "4", "--max-running-requests", "1",
        "--chunked-prefill-size", "2048",
    ]


def server_environment():
    return {
        **os.environ,
        "PYTHONHASHSEED": "42",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "SGLANG_MAMBA_CONV_DTYPE": "float16",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=30004)
    args = parser.parse_args(argv)
    return subprocess.call(server_command(args.model, args.port), env=server_environment())


if __name__ == "__main__":
    raise SystemExit(main())
