"""
    Purpose: shared CLI helpers — logging, main runner, CUDA check, validators
"""

import logging
import sys
from typing import Any, Callable, NoReturn


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for bench scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def run_main(main_fn: Callable[[], Any]) -> None:
    """Run main and exit with status 0 on success, 1 on exception."""
    try:
        main_fn()
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal error")
        sys.exit(1)
    sys.exit(0)


def require_cuda() -> None:
    """Raise RuntimeError if CUDA is not available."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark requires a GPU.")


def validate_nonempty(name: str, value: Any) -> None:
    """Raise ValueError if value is empty (None, empty str, etc.)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} must be non-empty")


def validate_positive_int(name: str, value: Any) -> None:
    """Raise ValueError if value is not a positive integer."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
