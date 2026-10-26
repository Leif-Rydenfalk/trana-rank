//! trana.rank — one deterministic, mergeable ranking over items and votes.
//!
//! Store additive statistics, apply every non-linear operation at read time:
//!   mean   = (M + a*c) / (N + a)          trust-weighted quality, prior smoothed
//!   volume = (N / (N + k))^g              breadth, saturating (sybil defence)
//!   div    = 1 / (1 + b*max(0, HHI - 1/S)) voter provenance diversity
//!   nov    = exp(-(now-ts0)/Hn) while n < n0
//!   score  = mean * volume * div * (1 + gn*nov)
//!
//! N = sum t(actor)*decay, M = sum dir*q*t(actor)*decay, HHI from (item,space) counts.
//! Everything but `mean` is additive; shards merge by taking the union of their top-k.
use serde_json::{json, Value};
use std::io::BufRead;
use std::time::Instant;

#[derive(Clone)]
struct Item {
    id: String,
    space: u32,
    ts: f64,
}

#[derive(Clone)]
struct Vote {
    item: u32,
    actor: u32,
    dir: f64,
    q: f64,
    ts: f64,
}

#[derive(Clone, Copy)]
struct Params {
    alpha: f64,
    kappa: f64,
    gamma: f64,
    beta: f64,
    gamma_n: f64,
    half_life_h: f64,
    new_half_life_h: f64,
    n0: f64,
    c: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { alpha: 6.0, kappa: 30.0, gamma: 0.7, beta: 1.2, gamma_n: 0.3,
                 half_life_h: 6.0, new_half_life_h: 2.0, n0: 25.0, c: 0.55 }
    }
}

fn main() {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).unwrap();
    let v: Value = serde_json::from_str(&line).unwrap_or_default();

    if v.get("bench").is_some() {
        bench(&v["bench"]);
        return;
    }

    let now = v["now"].as_f64().unwrap_or(1_792_974_000.0);
    let k = v["k"].as_u64().unwrap_or(10) as usize;
    let items: Vec<Item> = v["items"].as_array().map(|a| a.iter().map(|x| Item {
        id: x["id"].as_str().unwrap_or("").to_string(),
        space: x["space"].as_u64().unwrap_or(0) as u32,
        ts: x["ts"].as_f64().unwrap_or(now),
    }).collect()).unwrap_or_default();
    let votes: Vec<Vote> = v["votes"].as_array().map(|a| a.iter().map(|x| Vote {
        item: x["item"].as_u64().unwrap_or(0) as u32,
        actor: x["actor"].as_u64().unwrap_or(0) as u32,
        dir: x["dir"].as_f64().unwrap_or(1.0),
        q: x["q"].as_f64().unwrap_or(0.5),
        ts: x["ts"].as_f64().unwrap_or(now),
    }).collect()).unwrap_or_default();
    let trust: Vec<f64> = v["trust"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
    let actor_space: Vec<u32> = v["actor_space"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
    let n_spaces = v["n_spaces"].as_u64().unwrap_or(2000) as u32;

    let p = Params::default();
    let ranked = rank(&items, &votes, &trust, &actor_space, now, k, &p, n_spaces);
    let bucket = (now / 30.0).floor() as i64;
    let out: Vec<Value> = ranked.iter().map(|r| json!({
        "id": items[r.idx].id,
        "score": round(r.score, 6),
        "mean": round(r.mean, 6),
        "volume": round(r.volume, 6),
        "div": round(r.div, 6),
        "nov": round(r.nov, 6),
    })).collect();
    println!("{}", json!({"items": out, "bucket": bucket, "hash": hash(&ranked), "n": items.len()}));
}

fn round(x: f64, d: i32) -> f64 {
    let m = 10f64.powi(d);
    (x * m).round() / m
}

struct Scored { idx: usize, score: f64, mean: f64, volume: f64, div: f64, nov: f64 }

fn rank(items: &[Item], votes: &[Vote], trust: &[f64], actor_space: &[u32],
        now: f64, k: usize, p: &Params, n_spaces: u32) -> Vec<Scored> {
    let n = items.len();
    let mut n_eff = vec![0f64; n];
    let mut m_eff = vec![0f64; n];
    let mut n_raw = vec![0f64; n];
    let half = p.half_life_h * 3600.0;
    let s = n_spaces.max(1) as u64;

    // HHI keys: one per vote, (item, home space)
    let mut keys: Vec<u64> = Vec::with_capacity(votes.len());
    for v in votes {
        let idx = v.item as usize;
        if idx >= n { continue; }
        let t = *trust.get(v.actor as usize).unwrap_or(&0.0);
        let decay = (-(now - v.ts) / half).exp();
        n_eff[idx] += t * decay;
        m_eff[idx] += v.dir * v.q * t * decay;
        n_raw[idx] += 1.0;
        let sp = *actor_space.get(v.actor as usize).unwrap_or(&0) as u64;
        keys.push(idx as u64 * s + sp);
    }
    keys.sort_unstable();
    let mut hhi = vec![0f64; n];
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j] == keys[i] { j += 1; }
        let item = (keys[i] / s) as usize;
        let share = (j - i) as f64 / n_raw[item].max(1.0);
        hhi[item] += share * share;
        i = j;
    }

    let mut out: Vec<Scored> = Vec::with_capacity(n);
    for idx in 0..n {
        let mean = (m_eff[idx] + p.alpha * p.c) / (n_eff[idx] + p.alpha);
        let volume = (n_eff[idx] / (n_eff[idx] + p.kappa)).powf(p.gamma);
        let div = 1.0 / (1.0 + p.beta * (hhi[idx] - 1.0 / s as f64).max(0.0));
        let nov = if n_raw[idx] < p.n0 {
            (-(now - items[idx].ts) / (p.new_half_life_h * 3600.0)).exp()
        } else { 0.0 };
        let score = mean * volume * div * (1.0 + p.gamma_n * nov);
        out.push(Scored { idx, score, mean, volume, div, nov });
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.idx.cmp(&b.idx)));
    out.truncate(k.min(out.len()));
    out
}

fn hash(ranked: &[Scored]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for r in ranked {
        for b in r.idx.to_le_bytes().iter().chain(r.score.to_bits().to_le_bytes().iter()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{:016x}", h)
}

// ---------- synthetic benchmark: same generator as the reference prototype ----------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, n: u64) -> u64 { self.next() % n.max(1) }
}

fn bench(b: &Value) {
    let n_items = b["items"].as_u64().unwrap_or(1_000_000) as usize;
    let n_votes = b["votes"].as_u64().unwrap_or(10_000_000) as usize;
    let n_actors = b["actors"].as_u64().unwrap_or(200_000) as usize;
    let n_spaces = b["spaces"].as_u64().unwrap_or(2_000) as u32;
    let now = 1_792_974_000.0;
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let t0 = Instant::now();
    let items: Vec<Item> = (0..n_items).map(|i| Item {
        id: format!("i{}", i),
        space: rng.range(n_spaces as u64) as u32,
        ts: now - (-rng.f().max(1e-9).ln()) * 6.0 * 3600.0,
    }).collect();
    let trust: Vec<f64> = (0..n_actors).map(|_| { let x = rng.f(); x * x }).collect();
    let actor_space: Vec<u32> = (0..n_actors).map(|_| rng.range(n_spaces as u64) as u32).collect();
    let hot: Vec<bool> = (0..n_items).map(|_| rng.f() < 0.02).collect();
    let votes: Vec<Vote> = (0..n_votes).map(|_| {
        let mut item = rng.range(n_items as u64) as u32;
        if !hot[item as usize] { item = rng.range(n_items as u64) as u32; }
        Vote {
            item,
            actor: rng.range(n_actors as u64) as u32,
            dir: if rng.f() < 0.82 { 1.0 } else { -1.0 },
            q: rng.f() * 0.7 + 0.15,
            ts: now - (-rng.f().max(1e-9).ln()) * 3.0 * 3600.0,
        }
    }).collect();
    let gen_ms = t0.elapsed().as_millis();

    let p = Params::default();
    let t0 = Instant::now();
    let ranked = rank(&items, &votes, &trust, &actor_space, now, 50, &p, n_spaces);
    let rank_ms = t0.elapsed().as_millis();

    // shard the item range, rank each shard, merge top-k, compare with global
    let shards = 8usize;
    let t0 = Instant::now();
    let mut merged: Vec<Scored> = Vec::new();
    for s in 0..shards {
        let start = n_items * s / shards;
        let end = n_items * (s + 1) / shards;
        let sub_items = &items[start..end];
        let sub_votes: Vec<Vote> = votes.iter().filter(|v| (v.item as usize) >= start && (v.item as usize) < end)
            .map(|v| Vote { item: v.item - start as u32, ..v.clone() }).collect();
        let sub = rank(sub_items, &sub_votes, &trust, &actor_space, now, 50, &p, n_spaces);
        for r in sub { merged.push(Scored { idx: r.idx + start, ..r }); }
    }
    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.idx.cmp(&b.idx)));
    merged.truncate(50);
    let merge_ms = t0.elapsed().as_millis();
    let merge_ok = merged.len() == ranked.len() && merged.iter().zip(&ranked).all(|(a, b)| a.idx == b.idx && a.score.to_bits() == b.score.to_bits());

    println!("{}", json!({
        "bench": true, "items": n_items, "votes": n_votes, "actors": n_actors, "spaces": n_spaces,
        "gen_ms": gen_ms, "rank_ms": rank_ms, "merge8_ms": merge_ms, "merge_ok": merge_ok,
        "top": ranked.iter().take(5).map(|r| json!([items[r.idx].id, round(r.score, 4)])).collect::<Vec<_>>(),
        "hash": hash(&ranked)
    }));
}
