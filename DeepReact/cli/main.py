"""Command-line interface for DeepReact."""

import argparse
import sys
from pathlib import Path

from deepreact import __version__
from deepreact.config.parser import load_config
from deepreact.workflow.workflow import WorkflowManager


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `deepreact` CLI."""
    parser = argparse.ArgumentParser(
        prog="deepreact",
        description="Autonomous training of reactive deep learning potentials",
    )
    parser.add_argument(
        "--version", action="version", version=f"deepreact {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    # deepreact run config.yaml
    run_parser = subparsers.add_parser(
        "run", help="Execute a DeepReact workflow"
    )
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            print(
                f"Error: config file not found: {config_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        config = load_config(config_path)
        workflow = WorkflowManager(config)
        workflow.run()


if __name__ == "__main__":
    main()
