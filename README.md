# trana-rank

A distributed, mergeable ranking algorithm for Trana: one scoring primitive behind the feed, the
proposals, the work board and the leaderboard.

**The rule:** store additive statistics; apply every non-linear operation at read time. Trust,
decay, priors, saturation and diversity become read-time transforms over counters that only add,
so shards merge exactly, tiles are deterministic and edge-cacheable, and a vote is O(1).

- [DESIGN.md](DESIGN.md) — the model, the score, the distribution, the abuse model, the rollout.
- `src/main.rs` — the cell: one JSON line in, one JSON line out; `{"bench": {...}}` runs the
  synthetic benchmark. Published on ce-net as `trana.rank`.
- `cell.toml` — name and the shape of the one line in and out.

Measured on 1M items / 10M votes (Rust): full rank 0.86 s; 8-shard merge identical to global (same
score bits); deterministic hash stable. Attack checks ship in the cell: `{"attack":"sybil","fake":5000}` takes a rank-100 item to outside
the top 200; `{"attack":"brigade","fake":300,"trust":0.9}` shows the diversity term halving a
brigade's score (0.82 → 0.40) without unseating it — a tax, not a veto.

MIT.
