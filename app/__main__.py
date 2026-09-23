"""``python -m app demo``: run the offline demo (see app/demo.py)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description="llm-gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="offline walkthrough of every feature (no key, no network)")
    demo.add_argument("--config-dir", default=None, help="demo config directory (default: examples/demo)")
    demo.add_argument("--verbose", action="store_true", help="also print the gateway's own log lines")
    args = parser.parse_args(argv)

    from .demo import main as demo_main

    return demo_main(args.config_dir, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
