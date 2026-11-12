#!/usr/bin/env python3
"""Paired comparisons and an illustrative rank for the hardening runs (stdlib only).

Rank rule (as specified for this gym): sum over experiments and metrics of
experiment weight x |coefficient| x improvement over the reference in the
metric's better direction / population std of that metric across the cohort.
The cohort here is the set of methods scored in the variant and shortcut
passes plus the reference; future normalisers are unknown, so magnitudes are
illustrative.  Seed-level noise is the standard error of the mean difference in
per-controller success: paired by seed for snapshots of the same training runs,
unpaired otherwise.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ref_cfg = json.loads((HERE.parents[1] / "scoring" / "reference.json").read_text())["experiments"]
EXPS = list(ref_cfg)
PAIRED = {"spiking_eprop_4000ep", "spiking_eprop_500ep", "spiking_eprop_untrained"}


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def sample_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def collect():
    rows = {}
    var = json.loads((R / "eval_variants.json").read_text())["test"]["scores"]
    for e in EXPS:
        for m, v in var["experiments"][e]["results"].items():
            rows.setdefault(m, {})[e] = dict(v, per_controller=var["diagnostics"].get(m, {}).get(e, {}).get("per_controller_success"))
    sc = json.loads((R / "shortcuts.json").read_text())
    for case, res in sc.items():
        test = res.get("test", {}).get("scores")
        if not test:
            continue
        for m, v in test["experiments"][EXPS[0]]["results"].items():
            if m in ("random_policy", "always_forward"):
                continue
            name = f"shortcut:{case}" + ("" if m == "spiking_eprop" else f":{m}")
            for e in EXPS:
                rows.setdefault(name, {})[e] = dict(test["experiments"][e]["results"][m], per_controller=None)
    return rows


def main():
    rows = collect()
    ref = rows["spiking_eprop"]
    scales = {}
    for e in EXPS:
        for metric in ref_cfg[e]["metrics"]:
            vals = [rows[m][e][metric] for m in rows if e in rows[m]]
            scales[(e, metric)] = sd(vals) or 1.0
    out = {"scales": {f"{e}/{k}": v for (e, k), v in scales.items()}, "methods": {}}
    for m, r in rows.items():
        rank, deltas, noise = 0.0, {}, {}
        for e in EXPS:
            w = ref_cfg[e]["weight"]
            for metric, spec in ref_cfg[e]["metrics"].items():
                sign = 1.0 if spec["higher_is_better"] else -1.0
                d = sign * (r[e][metric] - ref[e][metric])
                deltas[f"{e}/{metric}"] = round(d, 4)
                rank += w * abs(spec["coefficient"]) * d / scales[(e, metric)]
            a, b = r[e].get("per_controller"), ref[e]["per_controller"]
            if a and b and m != "spiking_eprop":
                if m in PAIRED:
                    diffs = [x - y for x, y in zip(a, b)]
                    se = sample_sd(diffs) / math.sqrt(len(diffs))
                else:
                    se = math.sqrt(sample_sd(a) ** 2 / len(a) + sample_sd(b) ** 2 / len(b))
                diff = sum(a) / len(a) - sum(b) / len(b)
                noise[e] = {"success_diff": round(diff, 4), "se": round(se, 4),
                            "z": round(diff / se, 2) if se > 0 else None,
                            "pairing": "same seeds" if m in PAIRED else "independent"}
        out["methods"][m] = {"rank_vs_reference": round(rank, 4), "directed_deltas": deltas, "seed_noise": noise}
    (R / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    for m, v in sorted(out["methods"].items(), key=lambda kv: -kv[1]["rank_vs_reference"]):
        print(f"{m:55s} rank {v['rank_vs_reference']:+.3f}  " +
              "  ".join(f"{e[:10]}:{n['success_diff']:+.3f}(z={n['z']})" for e, n in v["seed_noise"].items()))


if __name__ == "__main__":
    main()
