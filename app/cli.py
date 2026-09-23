"""``llm-gateway`` command line (also ``python -m app``).

    llm-gateway serve [--host 127.0.0.1] [--port 8000] [--config-dir DIR] [--reload]
    llm-gateway demo [--config-dir examples/demo] [--verbose]
    llm-gateway check-config [--config-dir DIR] [--strict] [--json]
    llm-gateway keys generate --name my-app --rpm 120 --budget 25
    llm-gateway keys hash sk-gw-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from . import __version__


def load_env_file(path: str) -> list[str]:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding it.

    A tiny, dependency-free .env reader: blank lines and ``#`` comments are
    skipped, an optional ``export`` prefix and surrounding quotes are removed.
    Returns the names that were set.
    """
    loaded: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if name and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded


def _load_env(env_file: Optional[str]) -> Optional[int]:
    """Load ``env_file`` (or ./.env when present). Returns an exit code on error."""
    path = env_file or ".env"
    if os.path.exists(path):
        names = load_env_file(path)
        print(f"loaded {len(names)} variable(s) from {path}", file=sys.stderr)
    elif env_file:
        print(f"error: env file not found: {env_file}", file=sys.stderr)
        return 2
    return None


def _settings(config_dir: Optional[str]):
    from .config import Settings

    return Settings.from_env(config_dir=config_dir)


def _print_problems(problems, out=print) -> tuple[int, int]:
    from .checks import summarize

    for problem in problems:
        out(f"  {problem}")
    return summarize(problems)


def cmd_check_config(args: argparse.Namespace) -> int:
    from .checks import validate_config

    failure = _load_env(args.env_file)
    if failure is not None:
        return failure
    settings = _settings(args.config_dir)
    problems = validate_config(settings)
    errors, warnings = sum(p.level == "error" for p in problems), sum(p.level == "warning" for p in problems)
    failed = errors > 0 or (args.strict and warnings > 0)
    if args.json:
        print(json.dumps({
            "config_dir": settings.config_dir,
            "ok": not failed,
            "errors": errors,
            "warnings": warnings,
            "problems": [p.to_dict() for p in problems],
        }, indent=2))
        return 1 if failed else 0
    print(f"Checking config in {os.path.abspath(settings.config_dir)}")
    for label, path in (("providers", settings.providers_file), ("routing", settings.routing_file),
                        ("keys", settings.keys_file), ("pricing", settings.pricing_file)):
        print(f"  {label:<10}{path}")
    print(f"  {'embedder':<10}{settings.embed_model}")
    if problems:
        print()
        _print_problems(problems)
    print()
    verdict = "FAILED" if failed else "OK"
    print(f"{verdict}: {errors} error(s), {warnings} warning(s)" + (" (--strict)" if args.strict else ""))
    return 1 if failed else 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .checks import validate_config

    failure = _load_env(args.env_file)
    if failure is not None:
        return failure
    if args.config_dir:
        os.environ["CONFIG_DIR"] = args.config_dir
    settings = _settings(args.config_dir)
    if not os.path.exists(settings.providers_file):
        print(
            f"error: {settings.providers_file} not found. Run from the directory holding providers.yaml, "
            "routing.yaml, keys.yaml and pricing.json, or pass --config-dir (or set CONFIG_DIR).",
            file=sys.stderr,
        )
        return 2
    problems = validate_config(settings)
    if problems:
        print("config check:")
        errors, _ = _print_problems(problems)
        if errors and args.strict:
            print("refusing to start with config errors (--strict)", file=sys.stderr)
            return 1

    import uvicorn

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as demo_main

    return demo_main(args.config_dir, verbose=args.verbose)


def cmd_keys(args: argparse.Namespace) -> int:
    from .keys import run_key_command

    args.command = args.key_command
    return run_key_command(args)


def build_parser() -> argparse.ArgumentParser:
    from .keys import add_key_commands

    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="OpenAI-compatible LLM gateway: caching, routing, failover, limits and cost accounting.",
    )
    parser.add_argument("--version", action="version", version=f"llm-gateway {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    serve = sub.add_parser("serve", help="run the gateway (uvicorn)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 in Docker)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config-dir", default=None, help="directory with the four config files (default: CONFIG_DIR or .)")
    serve.add_argument("--env-file", default=None, help="load variables from this file (default: ./.env if present)")
    serve.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    serve.add_argument("--strict", action="store_true", help="refuse to start if check-config finds errors")
    serve.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo", help="offline walkthrough of every feature (no key, no network)")
    demo.add_argument("--config-dir", default=None, help="demo config directory (default: examples/demo)")
    demo.add_argument("--verbose", action="store_true", help="also print the gateway's own log lines")
    demo.set_defaults(func=cmd_demo)

    check = sub.add_parser("check-config", help="validate the config files and their cross-references")
    check.add_argument("--config-dir", default=None, help="directory with the four config files (default: CONFIG_DIR or .)")
    check.add_argument("--strict", action="store_true", help="treat warnings as errors")
    check.add_argument("--json", action="store_true", help="machine-readable output")
    check.add_argument("--env-file", default=None, help="load variables from this file (default: ./.env if present)")
    check.set_defaults(func=cmd_check_config)

    keys = sub.add_parser("keys", help="mint or hash gateway virtual keys")
    key_sub = keys.add_subparsers(dest="key_command", required=True, metavar="ACTION")
    add_key_commands(key_sub)
    keys.set_defaults(func=cmd_keys)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

