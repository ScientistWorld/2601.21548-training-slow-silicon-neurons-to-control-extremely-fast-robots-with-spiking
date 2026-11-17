# Targets

All three experiments report the same four evaluator-derived metrics:
`success_rate` (ranked, higher), `seed_worst_success` and `hit_rate`
(diagnostics, higher), and `controller_units` (diagnostic capacity measurement).

| Experiment | Scientific target | Capacity constraint | Weight |
|---|---|---:|---:|
| velocity_range_extreme | reliable interception across 0.5–2.0 m/s | fixed 1020 units | 0.2 |
| single_core_extreme | reliable interception at restricted capacity | fixed 256 units | 0.3 |
| sensor_robustness | reliable interception under richer launches, state noise and latency | fixed 1020 units | 0.5 |

All ten assigned instances must be present and valid. A failed constraint
sets every success metric to zero in every experiment. Final measured reference
and fixed baselines are in `reference.json`; original paper numbers remain
unchanged under `_paper_reported` because the local software scale differs.
