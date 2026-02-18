#!/usr/bin/env python
"""
arXiv RAG v1 - Streamlit UI

Runs the Streamlit search demo interface.

Usage:
    python scripts/05_ui.py
    python scripts/05_ui.py --port 8501
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run arXiv RAG Streamlit UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to bind (default: 8501)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("arXiv RAG v1 - Streamlit UI")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"URL: http://localhost:{args.port}")
    print("=" * 60)
    print()

    # Build streamlit command
    streamlit_app = project_root / "src" / "ui" / "streamlit_app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        str(args.port),
        "--server.address",
        "0.0.0.0",
    ]

    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    # Run streamlit
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
