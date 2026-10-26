"""Trana rank — reference implementation + benchmark.

Additive statistics at write time, decay and diversity at read time, so any shard can be
ranked alone and the results merged exactly (a mergeable top-k). Deterministic for a
quantised time bucket, so replicas agree byte for byte and edges can cache it.

    score(i) = [ S_i + alpha * prior_i ] / (1 + alpha) * div_i * (1 + gamma * nov_i)

    S_i    = sum over votes: dir * q * trust(actor) * exp(-(now - ts)/H)
    prior  = (pos + m * C) / (n + m)            Bayesian mean, C = global mean vote
    div    = 1 / (1 + beta * max(0, HHI_i - 1/S))
    nov    = exp(-(now - item_ts)/H_new)         only while n < n0

Everything that depends on the clock is applied at read time, so stored state is additive.
"""
import hashlib
import math
import time

import numpy as np

# --- parameters (one place, like the design skill's "make it yours") -------------------
P = dict(
    alpha=6.0,     # prior strength, in effective votes
    kappa=30.0,    # volume scale: how much breadth it takes to approach full score
    gamma=0.7,     # how hard breadth saturates
    beta=1.2,      # diversity penalty
    gamma_n=0.3,   # novelty weight
    half_life_h=6.0,
    new_half_life_h=2.0,
    n0=25,
    c=0.55,        # global prior mean: broadcast, never computed per shard
)


def gen(n_items=1_000_000, n_votes=10_000_000, n_actors=200_000, n_spaces=2_000, seed=7):
    rng = np.random.default_rng(seed)
    now = 1_792_974_000.0
    item_space = rng.integers(0, n_spaces, n_items)
    item_ts = now - rng.exponential(6 * 3600, n_items)
    actor_trust = rng.beta(2, 5, n_actors)          # most actors low trust, a few high
    actor_space = rng.integers(0, n_spaces, n_actors)  # where an actor mostly lives

    vi = rng.integers(0, n_items, n_votes)
    va = rng.integers(0, n_actors, n_votes)
    # votes concentrate on a few items, as they do in a real feed
    hot = rng.random(n_items) < 0.02
    vi = np.where(hot[vi], vi, rng.integers(0, n_items, n_votes))
    dir_ = np.where(rng.random(n_votes) < 0.82, 1.0, -1.0)
    q = rng.beta(2, 3, n_votes)                     # review quality of the reason
    vts = now - rng.exponential(3 * 3600, n_votes)
    return dict(now=now, n_items=n_items, item_space=item_space, item_ts=item_ts,
                actor_trust=actor_trust, actor_space=actor_space,
                vi=vi, va=va, dir=dir_, q=q, vts=vts)


def rank(d, now=None, k=50, shards=1, params=P):
    """Top-k ids. Every stored statistic is additive, so shards merge exactly."""
    now = now or d["now"]
    n_items = d["n_items"]
    half = params["half_life_h"] * 3600.0

    t = d["actor_trust"][d["va"]]
    decay = np.exp(-(now - d["vts"]) / half)
    n_eff = np.bincount(d["vi"], weights=t * decay, minlength=n_items)          # effective votes
    m_eff = np.bincount(d["vi"], weights=d["dir"] * d["q"] * t * decay, minlength=n_items)
    n_raw = np.bincount(d["vi"], minlength=n_items).astype(float)               # for novelty only

    # HHI over the voter's home space: sum over spaces of (share)^2
    S = int(d.get("n_spaces", 2000))
    space_of_vote = d["actor_space"][d["va"]].astype(np.int64)
    keys = d["vi"].astype(np.int64) * S + space_of_vote
    uniq, counts = np.unique(keys, return_counts=True)
    item_of = (uniq // S).astype(np.int64)
    nn = np.maximum(n_raw[item_of], 1.0)
    hhi = np.zeros(n_items)
    np.add.at(hhi, item_of, (counts / nn) ** 2)

    C = params["c"]
    mean = (m_eff + params["alpha"] * C) / (n_eff + params["alpha"])
    volume = (n_eff / (n_eff + params["kappa"])) ** params["gamma"]
    div = 1.0 / (1.0 + params["beta"] * np.maximum(0.0, hhi - 1.0 / max(S, 1)))
    nov = np.where(n_raw < params["n0"],
                   np.exp(-(now - d["item_ts"]) / (params["new_half_life_h"] * 3600)), 0.0)
    score = mean * volume * div * (1.0 + params["gamma_n"] * nov)

    if shards > 1:
        return _sharded(d, now, k, shards, params)

    idx = np.argpartition(-score, k)[:k]
    idx = idx[np.argsort(-score[idx])]
    return idx, score, dict(n_eff=n_eff, m_eff=m_eff, mean=mean, volume=volume, div=div, nov=nov)


def _sharded(d, now, k, shards, params):
    """Rank each shard independently and merge — must equal the global top-k."""
    parts = []
    for s in range(shards):
        sub = _slice(d, s, shards)
        idx, score, _ = rank(sub, now=now, k=k, shards=1, params=params)
        parts.append((sub["_offset"] + idx, score[idx]))
    ids = np.concatenate([p[0] for p in parts])
    sc = np.concatenate([p[1] for p in parts])
    order = np.argsort(-sc)[:k]
    return ids[order], None, dict(mode="merged")


def _slice(d, s, shards):
    """Contiguous item range [start, end) — the unit a compute shard owns."""
    n = d["n_items"]
    start = (n * s) // shards
    end = (n * (s + 1)) // shards
    m = (d["vi"] >= start) & (d["vi"] < end)
    out = dict(d)
    out["n_items"] = end - start
    out["item_space"] = d["item_space"][start:end]
    out["item_ts"] = d["item_ts"][start:end]
    out["vi"] = d["vi"][m] - start
    out["va"] = d["va"][m]
    out["dir"] = d["dir"][m]
    out["q"] = d["q"][m]
    out["vts"] = d["vts"][m]
    out["_offset"] = start
    return out


def digest(ids, score):
    return hashlib.sha256((ids.tobytes() + np.round(score, 9).tobytes())).hexdigest()[:16]


def sybil_test(d, victim=0, n_fake=5_000, trust=0.05):
    """5k fresh identities upvote one item: how far does it move without the defences?"""
    base_ids, base_score, _ = rank(d, k=20)
    dd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    dd["vi"] = np.concatenate([dd["vi"], np.full(n_fake, victim)])
    dd["va"] = np.concatenate([dd["va"], np.arange(n_fake) % len(dd["actor_trust"])])
    dd["dir"] = np.concatenate([dd["dir"], np.ones(n_fake)])
    dd["q"] = np.concatenate([dd["q"], np.full(n_fake, 0.1)])
    dd["vts"] = np.concatenate([dd["vts"], np.full(n_fake, d["now"])])
    ids, score, _ = rank(dd, k=20)
    pos = int(np.where(ids == victim)[0][0]) if victim in ids else None
    return pos, base_ids.tolist()[:5], ids.tolist()[:5]


if __name__ == "__main__":
    t0 = time.time()
    d = gen()
    d["n_spaces"] = 2000
    print(f"generated {d['n_items']:,} items / {len(d['vi']):,} votes in {time.time()-t0:.1f}s")

    t0 = time.time(); ids, score, parts = rank(d, k=50); t1 = time.time()
    print(f"full rank: {t1-t0:.2f}s   top5: {ids[:5].tolist()}")
    print(f"top score {score[ids[0]]:.3f}  median {np.median(score):.3f}  zero-score items {(score==0).mean():.1%}")

    ids2, score2, _ = rank(d, k=50)
    print("deterministic:", digest(ids, score[ids]) == digest(ids2, score2[ids2]))

    t0 = time.time(); mids, _, m = rank(d, k=50, shards=8); t1 = time.time()
    print(f"8 shards + merge: {t1-t0:.2f}s   identical to global: {np.array_equal(mids, ids)}")

    # incremental: 1% new votes on top items
    dd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    extra = 100_000
    dd["vi"] = np.concatenate([dd["vi"], np.random.default_rng(1).integers(0, d["n_items"], extra)])
    dd["va"] = np.concatenate([dd["va"], np.random.default_rng(2).integers(0, len(dd["actor_trust"]), extra)])
    dd["dir"] = np.concatenate([dd["dir"], np.ones(extra)])
    dd["q"] = np.concatenate([dd["q"], np.full(extra, 0.5)])
    dd["vts"] = np.concatenate([dd["vts"], np.full(extra, d["now"])])
    t0 = time.time(); rank(dd, k=50); print(f"re-rank after +{extra:,} votes: {time.time()-t0:.2f}s")

    pos, before, after = sybil_test(d)
    print(f"sybil: 5k fresh identities upvoting item 0 -> position {pos}, before {before[:3]}, after {after[:3]}")
