"""Command-line interface for dsh-copilot-sync.

Exit codes:
    0  success (all matched endpoints synced or already up to date)
    1  configuration error (bad source, bad target, duplicate names, ...)
    2  some endpoints failed (endpoint-level errors recorded)
    3  writing the target file failed
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import __version__
from .config import load_config, read_original_text, write_config_atomic
from .dsh_source import load_dsh_source
from .models import ModelSyncError, WriteError
from .sync import sync_config

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1
EXIT_PARTIAL_FAILURE = 2
EXIT_WRITE_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsh-copilot-sync",
        description=(
            "Offline sync of model lists from a dsh/ data folder into VS "
            "Code Copilot BYOK's chatLanguageModels.json, matching "
            "same-named custom endpoints. No network, no API keys."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dsh-copilot-sync --dsh-dir ~/.dsh --config ~/.vscode-chat/chatLanguageModels.json --all\n"
            "  dsh-copilot-sync --dsh-dir ~/.dsh --config chatLanguageModels.json --all --dry-run\n"
            "  dsh-copilot-sync --dsh-dir ~/.dsh --config chatLanguageModels.json --provider EndpointA --no-delete\n"
        ),
    )
    parser.add_argument(
        "--dsh-dir",
        required=True,
        metavar="PATH",
        help="path to the dsh data folder (source of model lists)",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="path to chatLanguageModels.json (target)",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--all",
        action="store_true",
        help="sync every same-named customendpoint in the target",
    )
    target.add_argument(
        "--provider",
        action="append",
        dest="providers",
        metavar="NAME",
        default=[],
        help="sync only the named target endpoint (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change without writing the file",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="only add new models; never remove models or settings entries",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed logs (unmatched endpoints, skipped models, ...)",
    )
    parser.add_argument(
        "--version", action="version", version=f"dsh-copilot-sync {__version__}"
    )
    return parser


def render_report(outcome, *, verbose: bool = False) -> List[str]:
    lines: List[str] = []
    for provider in outcome.providers:
        status = provider.status
        lines.append(f"[{status}] {provider.name}")
        if provider.added:
            lines.append(f"    added   ({len(provider.added)}): " + ", ".join(provider.added))
        if provider.removed:
            lines.append(f"    removed ({len(provider.removed)}): " + ", ".join(provider.removed))
        if provider.settings_keys_removed:
            lines.append(
                f"    settings keys removed ({len(provider.settings_keys_removed)}): "
                + ", ".join(provider.settings_keys_removed)
            )
        if provider.kept:
            lines.append(f"    kept    ({provider.kept}) with existing local config")
        if provider.skipped_deletion and provider.status == "skipped":
            lines.append("    deletion skipped: source read failed or model list is empty")
        if verbose and provider.skipped_model_ids:
            lines.append(
                f"    skipped new models ({len(provider.skipped_model_ids)}, no url): "
                + ", ".join(provider.skipped_model_ids)
            )
        for error in provider.errors:
            lines.append(f"    error: {error}")
    if verbose:
        for name in outcome.unmatched_source:
            lines.append(f"[skipped] source endpoint {name!r} has no same-named target customendpoint — not created")
    return lines


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provider_names: Optional[List[str]] = args.providers or None

    # 1. Read the source (config error -> exit 1, target never touched).
    try:
        source = load_dsh_source(args.dsh_dir)
    except ModelSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # 2. Read the target (config error -> exit 1, target never touched).
    try:
        config = load_config(args.config)
    except ModelSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # 3. Compute the sync (duplicate target names / unknown --provider -> exit 1).
    try:
        outcome = sync_config(
            config,
            source,
            allow_delete=not args.no_delete,
            provider_names=provider_names,
        )
    except ModelSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    report = render_report(outcome, verbose=args.verbose)

    if args.dry_run:
        print("Dry run - no files were written.")
        for line in report:
            print(line)
    else:
        try:
            result = write_config_atomic(
                args.config,
                outcome.config,
                original_text=read_original_text(args.config),
            )
        except WriteError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_WRITE_ERROR
        if result.written:
            print(f"Updated {args.config}")
        else:
            print(f"No changes required for {args.config}")
        for line in report:
            print(line)

    failed = [p for p in outcome.providers if not p.ok]
    changed = [p for p in outcome.providers if p.changed]
    total = len(outcome.providers)
    print(
        f"\nSummary: {total} endpoint(s) targeted, "
        f"{len(changed)} changed, {len(failed)} with errors, "
        f"{len(outcome.unmatched_source)} source endpoint(s) unmatched."
    )
    if failed:
        # Repeat the report on stderr so CI/pipe consumers see the problems.
        for line in report:
            print(line, file=sys.stderr)
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
