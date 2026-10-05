"""
app.py
------
Root entrypoint for cloud hosting (Render, Railway, Hugging Face Spaces)
and local development.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path so app package imports cleanly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import build_ui

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        footer_links=[],
        run_history=False,
    )
