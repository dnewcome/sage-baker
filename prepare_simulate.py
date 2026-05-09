"""Driver: generate a synthetic dataset from a registered scenario.

Mirrors the prepare_*.py pattern — produces training.parquet plus a
lineage.json, and additionally writes ground_truth.parquet for
evaluation use.

Usage::

    python prepare_simulate.py --scenario fuzzy_clickstream --output ./data_fuzzy/
    python prepare_simulate.py --scenario fuzzy_clickstream --output ./data_fuzzy/ \\
        --seed 7 --easy-mode

List scenarios::

    python prepare_simulate.py --list
"""
import argparse
import os
import shutil
import sys

# Repo root on sys.path so `import simulate` works whether this is run
# from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import describe_scenarios, get_scenario, list_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="scenario name (see --list)")
    parser.add_argument("--output", help="output directory; will be wiped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identified-fraction", type=float, default=None,
                        help="fraction of users who can ever be identified "
                             "(default per-scenario; fuzzy_clickstream defaults "
                             "to 0.0 = all anonymous)")
    parser.add_argument("--easy-mode", action="store_true",
                        help="reduce realism (force everyone identified + "
                             "always logged in; useful for sanity baselines)")
    parser.add_argument("--list", action="store_true",
                        help="list available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        for name, desc in describe_scenarios().items():
            print(f"{name}\n  {desc}\n")
        return

    if not args.scenario or not args.output:
        parser.error("--scenario and --output are required (or pass --list)")

    if args.scenario not in list_scenarios():
        parser.error(
            f"unknown scenario {args.scenario!r}; "
            f"available: {list_scenarios()}"
        )

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    scenario = get_scenario(args.scenario)
    overrides = {"easy_mode": args.easy_mode}
    if args.identified_fraction is not None:
        overrides["identified_user_fraction"] = args.identified_fraction
    result = scenario.generate(seed=args.seed, **overrides)
    result.write(args.output)

    print(f"wrote {args.output}/training.parquet     ({len(result.training)} rows × {len(result.training.columns)} cols)")
    print(f"wrote {args.output}/ground_truth.parquet ({len(result.ground_truth)} rows × {len(result.ground_truth.columns)} cols)")
    print(f"wrote {args.output}/lineage.json         (sha256: {result.lineage['dataset_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
