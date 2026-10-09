Offline value-function experiments on dev datasets (`s3://research/dq/collect2/`, `collect3/`, downloaded to
`scratch/c2_<B>`, `scratch/c3_<B>`). Run from the repo root (scripts read `scratch/` paths, e.g. `scratch/zone_geo.npz`,
`scratch/event_t.npy`). See LSTD_DQ_REPORT.md Part 3 in or-gymnax for results.
- tod.py: time-of-day / demand-rate features; nl.py: nonlinear count features
- fs2.py / fs5.py: estimate vs amount of pooled data (zone granularity, smooth spatial bases, busy time)
- fs7.py: LSTD(lambda_v) and gamma<1; fs8.py: time-block-varying average reward
- ev35.py: dev evaluation (own/pooled theta, traces, jackknife) at the A=0.35 pairs
