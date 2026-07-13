#!/usr/bin/env python3
"""Generate a launches.json from a launch.py descriptor file.

Usage:
    ./gen_launches_json.py launch.py              # writes launches.json next to launch.py
    ./gen_launches_json.py launch.py -o out.json
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from kernel_launch import save_launches


def load_launches(launch_path: Path) -> list:
    spec = importlib.util.spec_from_file_location("_launch_desc", launch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "LAUNCHES"):
        return mod.LAUNCHES
    raise AttributeError(f"{launch_path} must define LAUNCHES = [KernelLaunch(...)]")


def main():
    parser = argparse.ArgumentParser(description="Serialize a launch.py to launches.json")
    parser.add_argument("launch", type=Path, help="Path to launch.py")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: launches.json next to launch.py)")
    args = parser.parse_args()

    launch_path = args.launch.resolve()
    if not launch_path.exists():
        sys.exit(f"error: {launch_path} not found")

    launches = load_launches(launch_path)
    out_path = args.output.resolve() if args.output else launch_path.parent / "launches.json"
    save_launches(launches, str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
