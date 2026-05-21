#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import random
import re
from statistics import mean, stdev


ROUTE_RE = re.compile(
    r"route:\s*(?P<route>\d+)\s+"
    r"result:\s*(?P<result>\S+)\s+"
    r"time ticks:\s*(?P<ticks>\d+)\s+"
    r"waypoints:\s*(?P<waypoints>\d+)"
    r"(?:\s+goal_distance_m:\s*(?P<goal>[0-9.]+))?\s+"
    r"route_completion_percent:\s*(?P<completion>[0-9.]+)"
)


def parse_run_args(run_args_path):
    out = {"num_routes": None, "seed": None}
    if not os.path.exists(run_args_path):
        return out
    text = open(run_args_path, "r", encoding="utf-8", errors="ignore").read()
    for pat in (r"num_routes:\s*(\d+)", r"--num-routes\s+(\d+)"):
        m = re.search(pat, text)
        if m:
            out["num_routes"] = int(m.group(1))
            break
    for pat in (r"seed:\s*(\d+)", r"--seed\s+(\d+)"):
        m = re.search(pat, text)
        if m:
            out["seed"] = int(m.group(1))
            break
    return out


def parse_times(times_path):
    rows = []
    if not os.path.exists(times_path):
        return rows
    with open(times_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = ROUTE_RE.search(ln.strip())
            if not m:
                continue
            result = m.group("result")
            route_row = {
                "route_index": int(m.group("route")),
                "result": result,
                "ticks": int(m.group("ticks")),
                "waypoints": int(m.group("waypoints")),
                "goal_distance_m": float(m.group("goal")) if m.group("goal") else None,
                "completion_percent": float(m.group("completion")),
                "success": 1 if result == "reached" else 0,
                "collision": 1 if result == "collision" else 0,
                "timeout": 1 if "timeout" in result else 0,
            }
            rows.append(route_row)
    return rows


def wilson_ci(successes, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    margin = (z / den) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(values, iters=5000, alpha=0.05, seed=42):
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    sims = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sims.append(mean(sample))
    sims.sort()
    lo_i = int((alpha / 2.0) * iters)
    hi_i = int((1.0 - alpha / 2.0) * iters) - 1
    return (sims[max(0, lo_i)], sims[min(iters - 1, hi_i)])


def permutation_pvalue(a, b, iters=20000, seed=42):
    if not a or not b:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(mean(a) - mean(b))
    joined = list(a) + list(b)
    n = len(a)
    cnt = 0
    for _ in range(iters):
        rng.shuffle(joined)
        diff = abs(mean(joined[:n]) - mean(joined[n:]))
        if diff >= obs:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def parse_specs(spec_args):
    specs = []
    for raw in spec_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --spec '{raw}', expected LABEL=GLOB")
        label, pattern = raw.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError(f"Invalid --spec '{raw}', expected LABEL=GLOB")
        specs.append((label, pattern))
    return specs


def to_run_dirs(matches):
    out = []
    for p in matches:
        if os.path.isdir(p):
            out.append(os.path.abspath(p))
        elif os.path.isfile(p) and os.path.basename(p) == "run_args.txt":
            out.append(os.path.abspath(os.path.dirname(p)))
    return sorted(set(out))


def collect_runs(specs, require_complete=False, min_planned_routes=1):
    route_rows = []
    run_rows = []
    for label, pattern in specs:
        run_dirs = to_run_dirs(glob.glob(pattern))
        for run_dir in run_dirs:
            run_args = parse_run_args(os.path.join(run_dir, "run_args.txt"))
            times = parse_times(os.path.join(run_dir, "times.txt"))
            planned = run_args["num_routes"]
            observed = len(times)
            complete = bool(planned and observed >= planned)
            if planned is not None and planned < min_planned_routes:
                continue
            if require_complete and not complete:
                continue
            if observed == 0:
                continue

            comp_vals = [r["completion_percent"] for r in times]
            succ_vals = [r["success"] for r in times]
            coll_vals = [r["collision"] for r in times]
            timeout_vals = [r["timeout"] for r in times]
            run_rows.append(
                {
                    "label": label,
                    "run_dir": run_dir,
                    "seed": run_args["seed"],
                    "planned_routes": planned,
                    "observed_routes": observed,
                    "complete_run": int(complete),
                    "success_rate": mean(succ_vals),
                    "collision_rate": mean(coll_vals),
                    "timeout_rate": mean(timeout_vals),
                    "mean_completion_percent": mean(comp_vals),
                }
            )

            for r in times:
                route_rows.append(
                    {
                        "label": label,
                        "run_dir": run_dir,
                        "seed": run_args["seed"],
                        **r,
                    }
                )
    return route_rows, run_rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_group_summary(route_rows, run_rows):
    labels = sorted(set(r["label"] for r in run_rows))
    out = []
    for label in labels:
        rr = [r for r in route_rows if r["label"] == label]
        sr = [r for r in run_rows if r["label"] == label]
        n_routes = len(rr)
        n_runs = len(sr)
        succ = sum(r["success"] for r in rr)
        coll = sum(r["collision"] for r in rr)
        tout = sum(r["timeout"] for r in rr)
        comp = [r["completion_percent"] for r in rr]

        succ_lo, succ_hi = wilson_ci(succ, n_routes)
        coll_lo, coll_hi = wilson_ci(coll, n_routes)
        tout_lo, tout_hi = wilson_ci(tout, n_routes)
        comp_lo, comp_hi = bootstrap_mean_ci(comp)

        run_comp = [r["mean_completion_percent"] for r in sr]
        run_succ = [r["success_rate"] for r in sr]
        run_coll = [r["collision_rate"] for r in sr]
        run_tout = [r["timeout_rate"] for r in sr]

        out.append(
            {
                "label": label,
                "n_runs": n_runs,
                "n_routes": n_routes,
                "route_success_rate": succ / n_routes if n_routes else float("nan"),
                "route_success_ci95_lo": succ_lo,
                "route_success_ci95_hi": succ_hi,
                "route_collision_rate": coll / n_routes if n_routes else float("nan"),
                "route_collision_ci95_lo": coll_lo,
                "route_collision_ci95_hi": coll_hi,
                "route_timeout_rate": tout / n_routes if n_routes else float("nan"),
                "route_timeout_ci95_lo": tout_lo,
                "route_timeout_ci95_hi": tout_hi,
                "route_mean_completion_percent": mean(comp) if comp else float("nan"),
                "route_completion_ci95_lo": comp_lo,
                "route_completion_ci95_hi": comp_hi,
                "seed_mean_completion_percent": mean(run_comp) if run_comp else float("nan"),
                "seed_std_completion_percent": stdev(run_comp) if len(run_comp) >= 2 else 0.0,
                "seed_mean_success_rate": mean(run_succ) if run_succ else float("nan"),
                "seed_std_success_rate": stdev(run_succ) if len(run_succ) >= 2 else 0.0,
                "seed_mean_collision_rate": mean(run_coll) if run_coll else float("nan"),
                "seed_std_collision_rate": stdev(run_coll) if len(run_coll) >= 2 else 0.0,
                "seed_mean_timeout_rate": mean(run_tout) if run_tout else float("nan"),
                "seed_std_timeout_rate": stdev(run_tout) if len(run_tout) >= 2 else 0.0,
            }
        )
    return out


def build_pairwise_tests(route_rows, run_rows, labels):
    tests = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            a_runs = [r for r in run_rows if r["label"] == a]
            b_runs = [r for r in run_rows if r["label"] == b]
            use_seed_level = len(a_runs) >= 2 and len(b_runs) >= 2
            if use_seed_level:
                a_comp = [r["mean_completion_percent"] for r in a_runs]
                b_comp = [r["mean_completion_percent"] for r in b_runs]
                a_succ = [r["success_rate"] for r in a_runs]
                b_succ = [r["success_rate"] for r in b_runs]
            else:
                a_route = [r for r in route_rows if r["label"] == a]
                b_route = [r for r in route_rows if r["label"] == b]
                a_comp = [r["completion_percent"] for r in a_route]
                b_comp = [r["completion_percent"] for r in b_route]
                a_succ = [r["success"] for r in a_route]
                b_succ = [r["success"] for r in b_route]

            tests.append(
                {
                    "group_a": a,
                    "group_b": b,
                    "level": "seed" if use_seed_level else "route",
                    "delta_completion_percent": mean(a_comp) - mean(b_comp),
                    "p_completion_perm": permutation_pvalue(a_comp, b_comp),
                    "delta_success_rate": mean(a_succ) - mean(b_succ),
                    "p_success_perm": permutation_pvalue(a_succ, b_succ),
                }
            )
    return tests


def fmt_pct(x):
    if x != x:
        return "nan"
    return f"{x * 100.0:.2f}%"


def fmt_num(x):
    if x != x:
        return "nan"
    return f"{x:.4f}"


def print_report(group_summary, tests):
    print("=== Group Summary ===")
    for g in group_summary:
        print(
            f"[{g['label']}] runs={g['n_runs']} routes={g['n_routes']} "
            f"success={fmt_pct(g['route_success_rate'])} "
            f"(95%CI {fmt_pct(g['route_success_ci95_lo'])}~{fmt_pct(g['route_success_ci95_hi'])}) "
            f"collision={fmt_pct(g['route_collision_rate'])} "
            f"(95%CI {fmt_pct(g['route_collision_ci95_lo'])}~{fmt_pct(g['route_collision_ci95_hi'])}) "
            f"timeout={fmt_pct(g['route_timeout_rate'])} "
            f"(95%CI {fmt_pct(g['route_timeout_ci95_lo'])}~{fmt_pct(g['route_timeout_ci95_hi'])}) "
            f"ARC={g['route_mean_completion_percent']:.2f}% "
            f"(95%CI {g['route_completion_ci95_lo']:.2f}%~{g['route_completion_ci95_hi']:.2f}%)"
        )
        print(
            f"  seed-level: completion mean={g['seed_mean_completion_percent']:.2f}% std={g['seed_std_completion_percent']:.2f}, "
            f"success mean={fmt_pct(g['seed_mean_success_rate'])} std={fmt_num(g['seed_std_success_rate'])}"
        )
    if tests:
        print("\n=== Pairwise Significance (Permutation Test) ===")
        for t in tests:
            print(
                f"{t['group_a']} vs {t['group_b']} ({t['level']}-level): "
                f"delta_ARC={t['delta_completion_percent']:.2f}% p={t['p_completion_perm']:.4f}; "
                f"delta_success={t['delta_success_rate']:.4f} p={t['p_success_perm']:.4f}"
            )


def main():
    p = argparse.ArgumentParser(
        description="Aggregate route-level metrics and produce CI/variance/significance stats."
    )
    p.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Data spec in LABEL=GLOB form. Repeatable. Example: --spec wm='src/experiments/wm/automatic_control_*'",
    )
    p.add_argument(
        "--require-complete",
        action="store_true",
        help="Only include runs where observed routes >= planned num_routes.",
    )
    p.add_argument(
        "--min-planned-routes",
        type=int,
        default=1,
        help="Ignore runs whose planned num_routes is below this threshold (default: 1).",
    )
    p.add_argument(
        "--out-dir",
        default="src/experiments/analysis",
        help="Output directory for CSV artifacts.",
    )
    args = p.parse_args()

    specs = parse_specs(args.spec)
    route_rows, run_rows = collect_runs(
        specs,
        require_complete=args.require_complete,
        min_planned_routes=args.min_planned_routes,
    )
    if not route_rows:
        raise SystemExit("No valid runs found. Check --spec patterns and files.")

    group_summary = build_group_summary(route_rows, run_rows)
    labels = sorted(set(r["label"] for r in run_rows))
    tests = build_pairwise_tests(route_rows, run_rows, labels)

    route_csv = os.path.join(args.out_dir, "route_level.csv")
    run_csv = os.path.join(args.out_dir, "run_level.csv")
    group_csv = os.path.join(args.out_dir, "group_summary.csv")
    pair_csv = os.path.join(args.out_dir, "pairwise_tests.csv")

    write_csv(
        route_csv,
        route_rows,
        [
            "label",
            "run_dir",
            "seed",
            "route_index",
            "result",
            "ticks",
            "waypoints",
            "goal_distance_m",
            "completion_percent",
            "success",
            "collision",
            "timeout",
        ],
    )
    write_csv(
        run_csv,
        run_rows,
        [
            "label",
            "run_dir",
            "seed",
            "planned_routes",
            "observed_routes",
            "complete_run",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_completion_percent",
        ],
    )
    write_csv(
        group_csv,
        group_summary,
        [
            "label",
            "n_runs",
            "n_routes",
            "route_success_rate",
            "route_success_ci95_lo",
            "route_success_ci95_hi",
            "route_collision_rate",
            "route_collision_ci95_lo",
            "route_collision_ci95_hi",
            "route_timeout_rate",
            "route_timeout_ci95_lo",
            "route_timeout_ci95_hi",
            "route_mean_completion_percent",
            "route_completion_ci95_lo",
            "route_completion_ci95_hi",
            "seed_mean_completion_percent",
            "seed_std_completion_percent",
            "seed_mean_success_rate",
            "seed_std_success_rate",
            "seed_mean_collision_rate",
            "seed_std_collision_rate",
            "seed_mean_timeout_rate",
            "seed_std_timeout_rate",
        ],
    )
    write_csv(
        pair_csv,
        tests,
        [
            "group_a",
            "group_b",
            "level",
            "delta_completion_percent",
            "p_completion_perm",
            "delta_success_rate",
            "p_success_perm",
        ],
    )

    print_report(group_summary, tests)
    print("\nArtifacts:")
    print(f"- {os.path.abspath(route_csv)}")
    print(f"- {os.path.abspath(run_csv)}")
    print(f"- {os.path.abspath(group_csv)}")
    print(f"- {os.path.abspath(pair_csv)}")


if __name__ == "__main__":
    main()
