#!/usr/bin/env python3
"""Run staged DBpedia50 reproduction commands.

This runner keeps the server workflow reproducible without committing generated
checkpoints, sampled data, or full result folders.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


STAGES = {
    "s0": {"scale": "db50_s5000", "expected": {"train": 143, "valid": 13, "test": 13}},
    "s1": {"scale": "db50_s1000", "expected": {"train": 715, "valid": 78, "test": 78}},
    "s2": {"scale": "db50_s500", "expected": {"train": 1430, "valid": 169, "test": 169}},
    "s3": {"scale": "db50_s100", "expected": {"train": 7150, "valid": 884, "test": 884}},
    "s4": {"scale": "db50_s50", "expected": {"train": 14313, "valid": 1781, "test": 1781}},
    "s5": {"scale": "db50_s10", "expected": {"train": 71591, "valid": 8944, "test": 8944}},
}

DEFAULT_STEPS = ["sample", "train_uncond", "train_pattern", "test_baseline", "test_rerank"]
DEFAULT_RERANK_TEST_PROPORTIONS = [0.1, 0.25, 1.0]


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def format_proportions(proportions: list[float]) -> str:
    return ", ".join(str(proportion) for proportion in proportions)


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"\n$ {shell_join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def score_suffix(test_proportion: float, constrained: bool, test_count0: bool, rerank_k: int = 1, rerank_alpha: float = 0.5) -> str:
    suffix = f"test|{test_proportion}xtest_topk0_{constrained}_{test_count0}"
    if rerank_k > 1:
        suffix += f"_rerank{rerank_k}_alpha{rerank_alpha}"
    return suffix


def sampled_paths(args: argparse.Namespace, scale: str) -> dict[str, Path]:
    base = Path(args.data_root) / args.dataname
    prefix = f"{args.dataname}-{scale}-{args.max_answer_size}"
    return {
        split: base / f"{prefix}-{split}-a2q.jsonl"
        for split in ("train", "valid", "test")
    }


def checkpoint_path(args: argparse.Namespace, scale: str, epoch: int, condition: str) -> Path:
    return Path(args.checkpoint_root) / args.modelname / (
        f"{args.dataname}-{scale}-{args.max_answer_size}-{epoch}-{condition}.pth"
    )


def score_path(args: argparse.Namespace, scale: str, test_proportion: float, rerank_k: int = 1) -> Path:
    suffix = score_suffix(
        test_proportion=test_proportion,
        constrained=True,
        test_count0=False,
        rerank_k=rerank_k,
        rerank_alpha=args.rerank_alpha,
    )
    return Path(args.result_root) / args.modelname / (
        f"{args.dataname}-{scale}-{args.max_answer_size}-{args.pattern_epochs}-scores({suffix}).csv"
    )


def candidates_path(args: argparse.Namespace, scale: str, test_proportion: float) -> Path:
    suffix = score_suffix(
        test_proportion=test_proportion,
        constrained=True,
        test_count0=False,
        rerank_k=args.rerank_k,
        rerank_alpha=args.rerank_alpha,
    )
    return Path(args.result_root) / args.modelname / (
        f"{args.dataname}-{scale}-{args.max_answer_size}-{args.pattern_epochs}-candidates({suffix}).jsonl"
    )


def maybe_skip(label: str, paths: list[Path], force: bool) -> bool:
    if force:
        return False
    if paths and all(path.exists() for path in paths):
        print(f"\n# Skip {label}: existing outputs found")
        for path in paths:
            print(f"#   {path}")
        return True
    return False


def build_common_model_args(args: argparse.Namespace, scale: str) -> list[str]:
    return [
        "--modelname", args.modelname,
        "--data_root", args.data_root,
        "-d", args.dataname,
        "--scale", scale,
        "-a", str(args.max_answer_size),
        "--checkpoint_root", args.checkpoint_root,
        "--result_root", args.result_root,
    ]


def run_sample(args: argparse.Namespace, scale: str) -> None:
    paths = list(sampled_paths(args, scale).values())
    if maybe_skip("sample", paths, args.force):
        return
    Path(args.data_root, args.dataname).mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "akgr.sampling.sample_parallel",
        "-s", scale,
        "-a", str(args.max_answer_size),
        "-p", str(args.workers),
    ], args.dry_run)


def run_train_uncond(args: argparse.Namespace, scale: str) -> None:
    out = checkpoint_path(args, scale, args.uncond_epochs, "unconditional")
    if maybe_skip("train_uncond", [out], args.force):
        return
    run([
        sys.executable, "-m", "akgr.abduction_model.main",
        *build_common_model_args(args, scale),
        "--mode", "training",
        "--condition", "unconditional",
        "--overwrite_batchsize", str(args.batch_size),
        "--save_frequency", str(args.save_frequency),
        "--override_nepoch", str(args.uncond_epochs),
    ], args.dry_run)


def run_train_pattern(args: argparse.Namespace, scale: str) -> None:
    out = checkpoint_path(args, scale, args.pattern_epochs, "pattern")
    if maybe_skip("train_pattern", [out], args.force):
        return
    run([
        sys.executable, "-m", "akgr.abduction_model.main",
        *build_common_model_args(args, scale),
        "--mode", "training",
        "--condition", "pattern",
        "-r", str(args.uncond_epochs),
        "--overwrite_batchsize", str(args.batch_size),
        "--save_frequency", str(args.save_frequency),
        "--override_nepoch", str(args.pattern_epochs),
    ], args.dry_run)


def run_test_baseline(args: argparse.Namespace, scale: str) -> None:
    out = score_path(args, scale, args.baseline_test_proportion, rerank_k=1)
    if maybe_skip("test_baseline", [out], args.force):
        return
    run([
        sys.executable, "-m", "akgr.abduction_model.main",
        *build_common_model_args(args, scale),
        "--mode", "testing",
        "--condition", "pattern",
        "--tuning",
        "-r", str(args.pattern_epochs),
        "--test_split", "test",
        "--test_proportion", str(args.baseline_test_proportion),
        "--test_top_k", "0",
        "--overwrite_batchsize", str(args.batch_size),
        "--constrained", "True",
        "--rerank_k", "1",
    ], args.dry_run)


def run_test_rerank_once(args: argparse.Namespace, scale: str, test_proportion: float) -> None:
    score = score_path(args, scale, test_proportion, rerank_k=args.rerank_k)
    candidates = candidates_path(args, scale, test_proportion)
    if maybe_skip(f"test_rerank({test_proportion})", [score, candidates], args.force):
        return
    run([
        sys.executable, "-m", "akgr.abduction_model.main",
        *build_common_model_args(args, scale),
        "--mode", "testing",
        "--condition", "pattern",
        "--tuning",
        "-r", str(args.pattern_epochs),
        "--test_split", "test",
        "--test_proportion", str(test_proportion),
        "--test_top_k", "0",
        "--overwrite_batchsize", str(args.batch_size),
        "--constrained", "True",
        "--rerank_k", str(args.rerank_k),
        "--rerank_alpha", str(args.rerank_alpha),
        "--rerank_log_candidates",
    ], args.dry_run)


def run_test_rerank(args: argparse.Namespace, scale: str) -> None:
    for test_proportion in args.rerank_test_proportions:
        run_test_rerank_once(args, scale, test_proportion)


def print_stage_info(args: argparse.Namespace, scale: str) -> None:
    expected = STAGES[args.stage]["expected"]
    print(f"# Stage: {args.stage} ({scale})")
    print(f"# Expected rows: train={expected['train']}, valid={expected['valid']}, test={expected['test']}")
    print(f"# Steps: {', '.join(args.steps)}")
    print(f"# Epochs: unconditional={args.uncond_epochs}, pattern={args.pattern_epochs}")
    print(f"# Test: baseline proportion={args.baseline_test_proportion}, rerank proportions={format_proportions(args.rerank_test_proportions)}, rerank_k={args.rerank_k}")
    if args.dry_run:
        print("# Dry run: commands will be printed only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DBpedia50 staged reproduction commands.")
    parser.add_argument("--stage", choices=sorted(STAGES), default="s3")
    parser.add_argument("--steps", nargs="+", choices=DEFAULT_STEPS, default=DEFAULT_STEPS)
    parser.add_argument("--modelname", default="GPT2_6_act_nt")
    parser.add_argument("--dataname", default="DBpedia50")
    parser.add_argument("--data-root", default="./sampled_data/")
    parser.add_argument("--checkpoint-root", default="checkpoints/")
    parser.add_argument("--result-root", default="./results/")
    parser.add_argument("--max-answer-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--uncond-epochs", type=int, default=20)
    parser.add_argument("--pattern-epochs", type=int, default=40)
    parser.add_argument("--baseline-test-proportion", type=float, default=1.0)
    parser.add_argument("--rerank-test-proportion", type=float, default=None, help="deprecated single rerank proportion override")
    parser.add_argument("--rerank-test-proportions", nargs="+", type=float, default=None)
    parser.add_argument("--rerank-k", type=int, default=4)
    parser.add_argument("--rerank-alpha", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="rerun steps even when expected outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()
    if args.rerank_test_proportion is not None and args.rerank_test_proportions is not None:
        parser.error("use either --rerank-test-proportion or --rerank-test-proportions, not both")
    if args.rerank_test_proportions is None:
        if args.rerank_test_proportion is None:
            args.rerank_test_proportions = DEFAULT_RERANK_TEST_PROPORTIONS
        else:
            args.rerank_test_proportions = [args.rerank_test_proportion]
    return args


def main() -> None:
    args = parse_args()
    scale = STAGES[args.stage]["scale"]
    print_stage_info(args, scale)
    runners = {
        "sample": run_sample,
        "train_uncond": run_train_uncond,
        "train_pattern": run_train_pattern,
        "test_baseline": run_test_baseline,
        "test_rerank": run_test_rerank,
    }
    for step in args.steps:
        runners[step](args, scale)


if __name__ == "__main__":
    main()
