"""Shared paths and model identifiers for the project examples."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

QWEN3_VL_4B_Instruct = "Qwen/Qwen3-VL-4B-Instruct-FP8"
QWEN3_06B_Embed = "Qwen/Qwen3-Embedding-0.6B"
