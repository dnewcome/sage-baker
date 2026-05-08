"""autoresearch-style agent that iteratively improves a sage-baker plugin.

Inspired by karpathy/autoresearch — an LLM agent edits a small Python
file, runs training (~seconds on sonar-sized data), reads the metric,
keeps the change if better, reverts if worse, repeats. Cheap iteration
is the whole point.

Loop:
  1. Read program.md — human-edited prompt with constraints + strategy
  2. Read the plugin file the agent is allowed to edit
  3. Ask Claude for a complete new version
  4. Write it, syntax-check, run `make train`, parse validation_accuracy
  5. Compare to the best so far; `git checkout --` to revert if worse
  6. Loop until --budget-seconds or --max-iterations

Prereqs:
  ANTHROPIC_API_KEY in .env (the Makefile auto-loads it into the kernel)
  pip install -r requirements-agent.txt   # adds the anthropic SDK
  data already prepared (`make data-sonar`)
  the plugin file under git (the agent reverts via `git checkout --`)

Usage:
  python agent.py
  python agent.py --plugin src/plugins/default.py --max-iterations 10
  python agent.py --budget-seconds 600   # 10 min wall-clock cap
"""
import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ANTHROPIC_MODEL = "claude-sonnet-4-6"  # cheap-ish, fast, capable enough


def load_dotenv(path=".env"):
    """Best-effort .env loader — same shape Make uses (`-include .env`).

    Allows `python agent.py` to work without `make` in front. Doesn't
    override variables already set in the environment.
    """
    if not os.path.exists(path):
        return
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read(path):
    return Path(path).read_text()


def write(path, content):
    Path(path).write_text(content)


def revert(path):
    subprocess.run(["git", "checkout", "--", path], check=True)


def syntax_ok(source):
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def parse_metric(stdout, metric_name=None):
    """Parse the trainer's last `validation_<name>=<float>` line.

    If metric_name is given, look for that specific name. Otherwise find
    any validation_<name>=… (so the agent works for either classification
    or regression without configuration). Higher-is-better convention.
    """
    pattern = rf"{metric_name}=([\d.]+)" if metric_name else r"validation_\w+=([\d.]+)"
    matches = re.findall(pattern, stdout)
    return float(matches[-1]) if matches else None


def strip_fences(text):
    """Lenient — strip ```python ... ``` if the model added it despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:python|py)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def propose(client, program, plugin_path, plugin_src, history, best):
    history_summary = "\n".join(
        f"  iter {i + 1}: {m:.4f} ({'kept' if kept else 'reverted'})"
        for i, (_, m, kept) in enumerate(history[-5:])
    ) or "  (none yet — this is iteration 1)"

    best_str = f"{best:.4f}" if best > -float("inf") else "no successful runs yet"

    prompt = f"""{program}

# Current plugin source ({plugin_path}):
```python
{plugin_src}
```

# Recent experiments (most recent last)
{history_summary}

# Best metric so far: {best_str}

Output a COMPLETE new version of the plugin file. Plain Python source.
No markdown fences, no commentary, no diff format — just the file."""

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return strip_fences(msg.content[0].text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plugin", default="src/plugins/default.py",
                   help="file the agent is allowed to edit")
    p.add_argument("--program", default="program.md",
                   help="prompt with constraints + strategy hints")
    p.add_argument("--metric", default=None,
                   help="metric name to track (default: any validation_<name>=…)")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--budget-seconds", type=int, default=1800,
                   help="wall-clock cap (default 30 min)")
    args = p.parse_args()

    load_dotenv()  # so `python agent.py` works without `make` in front
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — add to .env first")
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("install agent deps first: pip install -r requirements-agent.txt")

    program = read(args.program)
    client = Anthropic()

    start = time.time()
    history = []  # [(proposal, metric, kept_bool)]
    best = -float("inf")

    for i in range(1, args.max_iterations + 1):
        elapsed = time.time() - start
        if elapsed > args.budget_seconds:
            print(f"budget exhausted at iteration {i}")
            break
        print(f"\n===== iteration {i}  best={best:.4f}  elapsed={int(elapsed)}s =====")

        plugin_src = read(args.plugin)

        try:
            proposal = propose(client, program, args.plugin, plugin_src, history, best)
        except Exception as e:
            print(f"  LLM call failed: {e}")
            history.append(("", -1.0, False))
            continue

        if not syntax_ok(proposal):
            print("  syntax error in proposal; reverting")
            history.append((proposal, -1.0, False))
            continue

        write(args.plugin, proposal)

        result = subprocess.run(
            ["make", "train"], capture_output=True, text=True, env=os.environ.copy()
        )
        if result.returncode != 0:
            print(f"  training failed (exit {result.returncode}); reverting")
            print(f"  last stderr: {result.stderr[-300:]}")
            revert(args.plugin)
            history.append((proposal, -1.0, False))
            continue

        metric = parse_metric(result.stdout, args.metric)
        if metric is None:
            print(f"  no '{args.metric}' in stdout; reverting")
            revert(args.plugin)
            history.append((proposal, -1.0, False))
            continue

        keep = metric > best
        print(f"  metric={metric:.4f} → {'KEEP' if keep else 'REVERT'}")
        if keep:
            best = metric
        else:
            revert(args.plugin)
        history.append((proposal, metric, keep))

    kept = sum(1 for _, _, k in history if k)
    print(f"\n===== done =====")
    print(f"  iterations: {len(history)} ({kept} kept, {len(history) - kept} reverted)")
    print(f"  best metric: {best:.4f}" if best > -float("inf") else "  no successful runs")
    print(f"  final plugin: {args.plugin} (whatever's currently checked out)")


if __name__ == "__main__":
    main()
