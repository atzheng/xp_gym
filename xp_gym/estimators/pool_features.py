"""State features for the pooled rideshare env (or_gymnax.rideshare_pool).

All features are aggregates over cars, evaluated at a reference time tau
(the time of the next request), so they describe the post-decision state.
"""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from importlib import resources as impresources

from xp_gym import data

# Bucket edges (seconds) for remaining-busy-time histograms.
TIME_EDGES = np.array([0, 120, 240, 360, 480, 600, 780, 960, 1200, 1500, 1800, 2400, 3600])
# Lookahead buckets (seconds) for when/where seats free up.
DROP_EDGES = np.array([0, 120, 300, 600])


def load_node_to_zone():
    zones = pd.read_parquet(impresources.files(data) / "taxi-zones.parquet")
    _, ids = np.unique(zones["zone"], return_inverse=True)
    return jnp.asarray(ids, dtype=jnp.int32), int(ids.max() + 1)


def _bucket(x):
    return jnp.searchsorted(jnp.asarray(TIME_EDGES[1:]), x, side="right")


def car_summary(waypoints, times, tau):
    """Per-car summaries. waypoints/times: (n_cars, 2m)."""
    n_cars, nwp = times.shape
    m = nwp // 2
    active = times > tau
    trip_active = active.reshape(n_cars, m, 2).any(-1)
    n_trips = trip_active.sum(-1)
    busy = jnp.maximum(times.max(-1) - tau, 0)  # time until car is empty
    drop_t = times.reshape(n_cars, m, 2)[..., 1]
    # time until next seat frees (first active dropoff)
    first_free = jnp.min(jnp.where(trip_active, drop_t - tau, jnp.iinfo(jnp.int32).max), -1)
    first_free = jnp.where(n_trips > 0, first_free, 0)
    # current/next location: next active waypoint, else last completed one
    nxt = jnp.argmin(jnp.where(active, times, jnp.iinfo(times.dtype).max), -1)
    last = jnp.argmax(times, -1)
    loc = jnp.where(active.any(-1), jnp.take_along_axis(waypoints, nxt[:, None], 1)[:, 0],
                    jnp.take_along_axis(waypoints, last[:, None], 1)[:, 0])
    final_loc = jnp.take_along_axis(waypoints, last[:, None], 1)[:, 0]
    # location where the first seat frees (first active dropoff)
    first_drop_idx = jnp.argmin(jnp.where(trip_active, drop_t, jnp.iinfo(times.dtype).max), -1)
    first_drop_loc = jnp.take_along_axis(
        waypoints.reshape(n_cars, m, 2)[..., 1], first_drop_idx[:, None], 1)[:, 0]
    return dict(n_trips=n_trips, busy=busy, first_free=first_free, loc=loc,
                final_loc=final_loc, first_drop_loc=first_drop_loc)


def feature_names(n_zones, m=2):
    nb = len(TIME_EDGES)
    names = [f"ntrips{k}" for k in range(m + 1)]
    for k in range(1, m + 1):
        names += [f"busy{k}_b{b}" for b in range(nb)]
    names += [f"firstfree{m}_b{b}" for b in range(nb)]
    names += [f"idle_z{z}" for z in range(n_zones)]
    names += [f"one_final_z{z}" for z in range(n_zones)]
    names += [f"full_drop_z{z}" for z in range(n_zones)]
    names += ["busy_sum", "busy_sq_sum", "firstfree_sum"]
    names += [f"drop_z{z}_b{b}" for b in range(len(DROP_EDGES)) for z in range(n_zones)]
    return names


def pool_features(waypoints, times, tau, node_to_zone, n_zones):
    """Superset feature vector (float32). See feature_names for layout."""
    n_cars, nwp = times.shape
    m = nwp // 2
    nb = len(TIME_EDGES)
    s = car_summary(waypoints, times, tau)
    nt = s["n_trips"]
    feats = [jnp.zeros(m + 1).at[nt].add(1.0)]
    bb = _bucket(s["busy"])
    for k in range(1, m + 1):
        feats.append(jnp.zeros(nb).at[bb].add((nt == k).astype(jnp.float32)))
    feats.append(jnp.zeros(nb).at[_bucket(s["first_free"])].add((nt == m).astype(jnp.float32)))
    feats.append(jnp.zeros(n_zones).at[node_to_zone[s["loc"]]].add((nt == 0).astype(jnp.float32)))
    feats.append(jnp.zeros(n_zones).at[node_to_zone[s["final_loc"]]].add((nt == 1).astype(jnp.float32)))
    feats.append(jnp.zeros(n_zones).at[node_to_zone[s["first_drop_loc"]]].add((nt == m).astype(jnp.float32)))
    busy = s["busy"].astype(jnp.float32) / 1000.0
    ff = s["first_free"].astype(jnp.float32) / 1000.0
    feats.append(jnp.stack([busy.sum(), (busy ** 2).sum(), jnp.where(nt == m, ff, 0).sum()]))
    # seats freeing up: active dropoffs by (zone, time-until-dropoff bucket)
    drop_wp = waypoints.reshape(n_cars, m, 2)[..., 1].reshape(-1)
    drop_dt = (times.reshape(n_cars, m, 2)[..., 1] - tau).reshape(-1)
    db = jnp.searchsorted(jnp.asarray(DROP_EDGES[1:]), drop_dt, side="right")
    feats.append(jnp.zeros(len(DROP_EDGES) * n_zones)
                 .at[db * n_zones + node_to_zone[drop_wp]].add((drop_dt > 0).astype(jnp.float32)))
    return jnp.concatenate(feats).astype(jnp.float32)


def reference_requests(events, K, seed=123):
    """Fixed sample of K requests from the event table (src, dest)."""
    idx = np.random.default_rng(seed).choice(len(events.src), K, replace=False)
    return jnp.asarray(np.asarray(events.src)[idx]), jnp.asarray(np.asarray(events.dest)[idx])


def match_costs(waypoints, times, tau, distances, ref_src, ref_dest, max_active_trips=2):
    """Insertion cost / feasibility / solo flag of every (reference request, car).
    Returns three (K, n_cars) arrays."""
    from or_gymnax.rideshare_pool import insert_and_optimize_trip

    def per_req(src, dest):
        def per_car(wp, t):
            _, _, cost, feas = insert_and_optimize_trip(
                distances, wp, t, src, dest, tau, max_active_trips)
            return cost, feas, jnp.all(t <= tau)
        return jax.vmap(per_car)(waypoints, times)

    return jax.vmap(per_req)(ref_src, ref_dest)


def match_features_from_costs(cost, feas, solo, distances, ref_src, ref_dest, thresholds,
                              profit_margin=1.0):
    """For each threshold h and reference request k: reward the greedy policy
    with threshold h would earn if k arrived now, and the number of eligible
    pooled cars. Layout matches match_feature_names."""
    d = distances[ref_src, ref_dest][:, None]
    outs = []
    for h in thresholds:
        elig = feas & (solo | (cost < d * (1 - h)))
        best = jnp.min(jnp.where(elig, cost, jnp.iinfo(cost.dtype).max), axis=1)
        rew = jnp.where(elig.any(1), d[:, 0] * (1 + profit_margin) - best, 0.0)
        outs += [rew.astype(jnp.float32), (elig & ~solo).sum(1).astype(jnp.float32)]
    return jnp.concatenate(outs)


def match_features(waypoints, times, tau, distances, ref_src, ref_dest, thresholds,
                   max_active_trips=2, profit_margin=1.0):
    c = match_costs(waypoints, times, tau, distances, ref_src, ref_dest, max_active_trips)
    return match_features_from_costs(*c, distances, ref_src, ref_dest, thresholds, profit_margin)


def match_feature_names(K, thresholds):
    names = []
    for h in thresholds:
        names += [f"rew_h{h}_k{k}" for k in range(K)]
        names += [f"npool_h{h}_k{k}" for k in range(K)]
    return names
