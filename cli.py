"""Command line interface for starting MAPPO training (standard library only, so it starts instantly)."""
import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

CHECKPOINT_DIR = "checkpoints"
WIDTH = 64


@dataclass
class RunConfig:
    checkpoint_path: Optional[str] = None  # None means train from scratch

    @property
    def resume(self) -> bool:
        return self.checkpoint_path is not None


def _checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path)
    return int(match.group(1)) if match else -1


def find_checkpoints(directory: str = CHECKPOINT_DIR) -> List[str]:
    if not os.path.isdir(directory):
        return []
    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".pt")
    ]
    return sorted(paths, key=lambda p: (_checkpoint_step(p), os.path.getmtime(p)))


def _line(char: str = "-") -> str:
    return char * WIDTH


def _banner() -> None:
    print()
    print(_line("="))
    print("MISFIT MARL TRAINING".center(WIDTH))
    print("Multi-agent PPO (MAPPO) - fixed-wing goal reaching".center(WIDTH))
    print(_line("="))


def _describe(path: str) -> str:
    size_mb = os.path.getsize(path) / 1e6
    modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    step = _checkpoint_step(path)
    step_text = f"step {step}" if step >= 0 else "custom"
    return f"{os.path.basename(path):<24} {step_text:<10} {size_mb:5.1f} MB  {modified}"


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)


def _choose_checkpoint(checkpoints: List[str]) -> Optional[str]:
    print()
    print("Available checkpoints")
    print(_line())
    for index, path in enumerate(checkpoints, start=1):
        marker = "  (latest)" if index == len(checkpoints) else ""
        print(f"  [{index}] {_describe(path)}{marker}")
    print(_line())
    print("  Enter a number, a file path, or 'b' to go back.")

    while True:
        answer = _ask("Checkpoint > ")
        if answer.lower() in ("b", "back"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(checkpoints):
            return checkpoints[int(answer) - 1]
        if os.path.isfile(answer):
            return answer
        print(f"  '{answer}' is not a valid choice.")


def _interactive_menu() -> RunConfig:
    _banner()
    while True:
        print()
        print("What would you like to do?")
        print(_line())
        print("  [1] Train from scratch")
        print("  [2] Load a prior checkpoint and continue training")
        print("  [q] Quit")
        print(_line())

        choice = _ask("Select > ").lower()
        if choice == "1":
            return RunConfig()
        if choice == "2":
            checkpoints = find_checkpoints()
            if not checkpoints:
                print(f"\n  No checkpoints found in '{CHECKPOINT_DIR}/'. Train from scratch first.")
                continue
            path = _choose_checkpoint(checkpoints)
            if path is not None:
                return RunConfig(checkpoint_path=path)
        elif choice in ("q", "quit", "exit"):
            sys.exit(0)
        else:
            print(f"  '{choice}' is not a valid choice.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MAPPO agents. Run without arguments for an interactive menu.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scratch",
        action="store_true",
        help="train from scratch without showing the menu",
    )
    group.add_argument(
        "--resume",
        metavar="PATH|latest",
        help="continue from a checkpoint file, or 'latest' for the newest one in checkpoints/",
    )
    return parser


def get_run_config(argv: Optional[List[str]] = None) -> RunConfig:
    args = _build_parser().parse_args(argv)

    if args.scratch:
        return RunConfig()

    if args.resume:
        if args.resume == "latest":
            checkpoints = find_checkpoints()
            if not checkpoints:
                sys.exit(f"error: no checkpoints found in '{CHECKPOINT_DIR}/'")
            return RunConfig(checkpoint_path=checkpoints[-1])
        if not os.path.isfile(args.resume):
            sys.exit(f"error: checkpoint not found: {args.resume}")
        return RunConfig(checkpoint_path=args.resume)

    if not sys.stdin.isatty():
        sys.exit("error: no terminal available for the menu; pass --scratch or --resume")
    return _interactive_menu()


def print_run_summary(config: RunConfig, start_step: int = 0) -> None:
    print()
    print(_line())
    if config.resume:
        print(f"Resuming from {config.checkpoint_path} (step {start_step})")
    else:
        print("Training from scratch")
    print(_line())
    print()
