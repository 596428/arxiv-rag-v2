#!/usr/bin/env python
"""
arXiv RAG v1 - API Server

Runs the FastAPI search API server.

Usage:
    python scripts/04_serve.py
    python scripts/04_serve.py --port 8000
    python scripts/04_serve.py --reload  # Development mode
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run arXiv RAG API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("arXiv RAG v1 - API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    print()
    print("API Documentation: http://localhost:{}/docs".format(args.port))
    print("Health Check: http://localhost:{}/health".format(args.port))
    print()

    import uvicorn

    uvicorn.run(
        "src.rag.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
