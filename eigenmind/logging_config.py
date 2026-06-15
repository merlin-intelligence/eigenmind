"""Central logging configuration for the eigenmind package.

Call :func:`setup_logging` once at process startup (done automatically by the
package ``__init__``). Every module then just does::

    import logging
    logger = logging.getLogger(__name__)

Output goes to *stderr*, which the systemd service captures in ``streamlit.log``
via ``StandardError=append:<path>``.
"""
from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger and silence chatty third-party libraries."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for noisy in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "urllib3",
        "httpx",
        "httpcore",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
