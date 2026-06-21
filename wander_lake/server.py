#!/usr/bin/env python3
"""WanderAI MCP-Gym server entrypoint.  Usage: python server.py --port 9100 --seed 0"""
import argparse
import os
import sys
from pathlib import Path

# Make both this dir (flat imports: wander_mcp/wander_adapter) and the repo root
# (the `wanderai` package) importable. On Fireworks the evaluator runs from the
# extracted tarball where `wanderai` is NOT pip-installed, so we resolve it by
# path relative to this file rather than relying on site-packages.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))           # wander_lake/  -> wander_mcp, wander_adapter
sys.path.insert(0, str(_HERE.parent))    # repo root     -> import wanderai

from wander_mcp import WanderMcp


def main():
    p = argparse.ArgumentParser(description="WanderAI MCP-Gym Server")
    p.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    server = WanderMcp(seed=args.seed)
    print(f"🚀 WanderAI MCP-Gym server on port {args.port} (seed {args.seed})")
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
