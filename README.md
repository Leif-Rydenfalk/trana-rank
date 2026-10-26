# trana-rank

A distributed, mergeable ranking algorithm for Trana: one scoring primitive behind the feed, the
proposals, the work board and the leaderboard.

**The rule:** store additive statistics; apply every non-linear operation at read time. Trust,
decay, priors, saturation and diversity become read-time transforms over counters that only add,
so shards merge exactly, tiles are deterministic and edge-cacheable, and a vote is O(1).

- [DESIGN.md](DESIGN.md) — the model, the score, the distribution, the abuse model, the rollout.
- `trana_rank.py` — reference implementation + benchmarks (numpy).

Measured on 1M items / 10M votes: full rank 0.85 s; 8-shard merge identical to global; deterministic
hash stable; 5k sybil identities do not enter the top 20; a single-space brigade is #1 without the
diversity term and out of the top 5 with it.

MIT.
