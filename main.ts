// roblox-up-and-coming-forecast-01 entry: Monte Carlo rank-diffusion forecaster.
//
// Model: each game's log-rank follows a random walk with drift. The drift is the
// game's recent log-rank velocity (last 48 h, recency-weighted, shrunk), which
// decays with a 24 h half-life as the window runs. Each hour, a step is drawn
// from the empirical distribution of past per-hour log-rank steps (so the heavy
// tails of real rank jumps are kept), plus an occasional larger jump. Ranks
// reflect at 1 and are capped at 130. Every simulated hour the games are
// ordered and the top 10 recorded; over many seeded simulations this gives
// P(game at place k at a random hour of the 72 h window). Probabilities are
// scaled to leave an unnamed share for games that appear out of nowhere
// (brand-new games do reach #1 on this chart). Deterministic: one fixed seed.
//
// Node 22, standard library only, no network. Reads $MAKEFUN_INPUT_DIR
// (default /input) history.ndjson + task.json, writes
// $MAKEFUN_OUTPUT_DIR/forecast.json (default /output).

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- tuned parameters (see NOTES.md; backtested on the practice window and
// --- on two internal train/test splits of the chart history) ---
// SIMS raised 1500 -> 3000 after the local grade: the forecast is a Monte
// Carlo average, so doubling the simulations halves its Monte Carlo error
// (practice score -34.660 -> -34.558, same hits/agreement) while the run
// stays ~7 s, far inside the 300 s limit.
//
// v2: parameters are overridable from the environment for backtesting
// (MF_*); the shipped defaults are the chosen v2 values.
// v2 changes (see NOTES.md): DRIFT_SHRINK 0.75 -> 1.0 (the recent
// log-rank velocity is a real signal: unshrunk drift won on the practice
// window and both internal splits) and MIN_P 0.0015 -> 0.0045 (dropping
// the tiniest probability rows concentrates mass where it counts).
// Occupancy blending and sharpening were tested and REJECTED: a pure
// occupancy table scores -55 on the practice window, blends hurt all
// three test beds, and sharpening buys practice points only by losing
// on split 1 (confident-wrong is punished hard by the log score).
const SIMS = 3000;
const HOURS = 72;
const SEED = 12345;
const SIGMA = Number(process.env.MF_SIGMA ?? 0.11); // scales the empirical per-hour log-rank steps
const DRIFT_SHRINK = Number(process.env.MF_DRIFT_SHRINK ?? 1.0); // shrink recent log-rank velocity toward 0
const DRIFT_HALF_LIFE_H = 24; // drift decays over the window
const DRIFT_CAP = 0.05; // max |log-rank velocity| per hour
const DRIFT_WINDOW_H = 48; // lookback for velocity estimation
const JUMP_P = 0.02; // per hour, per game: extra jump probability
const JUMP_SD = 0.5; // extra jump size in log-rank
const UNNAMED = Number(process.env.MF_UNNAMED ?? 0.15); // share of each place left for unlisted games
const MIN_P = Number(process.env.MF_MIN_P ?? 0.0045); // smallest named probability
const OCC_W = Number(process.env.MF_OCC_W ?? 0); // blend weight of short-window place occupancy (0 = off)
const OCC_WINDOW_H = Number(process.env.MF_OCC_WINDOW_H ?? 48); // occupancy lookback, hours
const SHARPEN = Number(process.env.MF_SHARPEN ?? 1); // exponent applied to blended probabilities (1 = off)
const MAX_ROWS_PER_PLACE = 100;
const MAX_RANK = 110; // simulation cap (chart is ~97-99 games)
const CANDIDATE_DAYS = 7; // games unseen for longer are dropped

interface Row {
  rank: number;
  universe_id: number;
  name: string;
  playing: number;
}

interface Snapshot {
  at: string;
  up_and_coming: Row[];
  details?: Record<string, { created?: string }> | null;
}

interface Task {
  places: number;
  output: string;
  window?: { from: string; until: string };
}

interface Candidate {
  id: number;
  y0: number;
  drift: number;
  exitAtMs: number | null; // created + 28d: the game leaves the chart then
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function (): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function main(): void {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
  const places = task.places;
  const hist = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((l) => JSON.parse(l) as Snapshot);
  hist.sort((a, b) => (a.at < b.at ? -1 : 1));
  const lastAt = new Date(hist[hist.length - 1]!.at).getTime();
  const lastChart = hist[hist.length - 1]!.up_and_coming;
  const lastIds = new Set(lastChart.map((r) => r.universe_id));

  // --- per-game observation series ---
  const games = new Map<number, { name: string; obs: Array<{ t: number; rank: number }> }>();
  const created = new Map<number, number>(); // universe_id -> created ms
  for (const s of hist) {
    const t = new Date(s.at).getTime();
    for (const r of s.up_and_coming) {
      let g = games.get(r.universe_id);
      if (g === undefined) {
        g = { name: r.name, obs: [] };
        games.set(r.universe_id, g);
      }
      g.obs.push({ t, rank: r.rank });
    }
    if (s.details) {
      for (const [idStr, d] of Object.entries(s.details)) {
        if (d && d.created && !created.has(Number(idStr))) {
          created.set(Number(idStr), new Date(d.created).getTime());
        }
      }
    }
  }

  // --- empirical per-sqrt-hour log-rank steps, by rank band ---
  const stepsTop: number[] = [];
  const stepsMid: number[] = [];
  const stepsLow: number[] = [];
  for (let i = 1; i < hist.length; i++) {
    const dt = (new Date(hist[i]!.at).getTime() - new Date(hist[i - 1]!.at).getTime()) / 3600000;
    if (dt <= 0) continue;
    const prev = new Map(hist[i - 1]!.up_and_coming.map((r) => [r.universe_id, r.rank] as const));
    for (const r of hist[i]!.up_and_coming) {
      const p = prev.get(r.universe_id);
      if (p === undefined) continue;
      const step = (Math.log(r.rank) - Math.log(p)) / Math.sqrt(dt);
      if (p <= 10) stepsTop.push(step);
      else if (p <= 30) stepsMid.push(step);
      else stepsLow.push(step);
    }
  }
  const allSteps = stepsTop.concat(stepsMid, stepsLow);
  const pick = (arr: number[]): number[] => (arr.length > 0 ? arr : allSteps);

  // --- candidates: games seen recently or on the last chart ---
  const cutoff = lastAt - CANDIDATE_DAYS * 86400000;
  const cand: Candidate[] = [];
  for (const [id, g] of games) {
    const lastObs = g.obs[g.obs.length - 1]!;
    const inLast = lastIds.has(id);
    if (lastObs.t < cutoff && !inLast) continue;
    const ageH = (lastAt - lastObs.t) / 3600000;
    let y0 = Math.log(lastObs.rank);
    if (!inLast) y0 = Math.log(Math.min(MAX_RANK, lastObs.rank + ageH * 0.5));
    // recent log-rank velocity, recency-weighted, shrunk and capped
    let num = 0;
    let den = 0;
    const recent = g.obs.filter((o) => o.t >= lastAt - DRIFT_WINDOW_H * 3600000);
    for (let i = 1; i < recent.length; i++) {
      const dt = (recent[i]!.t - recent[i - 1]!.t) / 3600000;
      if (dt <= 0) continue;
      const v = (Math.log(recent[i]!.rank) - Math.log(recent[i - 1]!.rank)) / dt;
      const w = 1 / (1 + (lastAt - recent[i]!.t) / 3600000 / 12);
      num += w * v;
      den += w;
    }
    let drift = den > 0 ? (num / den) * DRIFT_SHRINK : 0;
    drift = Math.max(-DRIFT_CAP, Math.min(DRIFT_CAP, drift));
    // age-out: the chart only lists games published in the last 28 days
    let exitAtMs: number | null = null;
    const cr = created.get(id);
    if (cr !== undefined) exitAtMs = cr + 28 * 86400000;
    if (exitAtMs !== null && exitAtMs <= lastAt) continue; // already aged out
    cand.push({ id, y0, drift, exitAtMs });
  }
  // deterministic candidate order
  cand.sort((a, b) => a.id - b.id);

  // --- Monte Carlo over the window ---
  const rng = mulberry32(SEED);
  const n = cand.length;
  const counts = new Map<number, Float64Array>();
  for (const c of cand) counts.set(c.id, new Float64Array(places + 1));
  const ys = new Float64Array(n);
  const order = Array.from({ length: n }, (_, i) => i);
  const windowMs = HOURS * 3600000;
  for (let s = 0; s < SIMS; s++) {
    for (let i = 0; i < n; i++) ys[i] = cand[i]!.y0;
    for (let h = 1; h <= HOURS; h++) {
      const decay = Math.pow(0.5, (h - 1) / DRIFT_HALF_LIFE_H);
      const tMs = lastAt + h * 3600000;
      for (let i = 0; i < n; i++) {
        const c = cand[i]!;
        let dy = c.drift * decay;
        const y = ys[i]!;
        const band = y < Math.log(11) ? pick(stepsTop) : y < Math.log(31) ? pick(stepsMid) : pick(stepsLow);
        dy += (SIGMA / 0.1) * band[(rng() * band.length) | 0]!;
        if (rng() < JUMP_P) dy += gauss(rng) * JUMP_SD;
        let ny = y + dy;
        if (ny < 0) ny = -ny; // reflect at rank 1
        if (ny > Math.log(130)) ny = Math.log(130);
        ys[i] = ny;
      }
      // order games by latent log-rank; skip games that aged out this hour
      order.sort((a, b) => ys[a]! - ys[b]!);
      let placed = 0;
      for (let k = 0; k < n && placed < places; k++) {
        const c = cand[order[k]!]!;
        if (c.exitAtMs !== null && tMs >= c.exitAtMs) continue;
        counts.get(c.id)![placed + 1]! += 1;
        placed++;
      }
      void windowMs;
    }
  }
  const totalSamples = SIMS * HOURS;

  // --- short-window place occupancy (v2 blend component) ---
  // For each place, the fraction of recent reads (last OCC_WINDOW_H hours)
  // in which each game held that place. A pure occupancy forecast scores
  // badly over 72 h (-55 on the practice window: the chart churns), but a
  // small blend weight sharpens the diffusion forecast toward games that
  // have actually held each place recently.
  const occCounts: Array<Map<number, number>> = Array.from({ length: places + 1 }, () => new Map());
  if (OCC_W > 0) {
    const occCutoff = lastAt - OCC_WINDOW_H * 3600000;
    for (const s of hist) {
      if (new Date(s.at).getTime() < occCutoff) continue;
      for (const r of s.up_and_coming) {
        if (r.rank < 1 || r.rank > places) continue;
        const m = occCounts[r.rank]!;
        m.set(r.universe_id, (m.get(r.universe_id) ?? 0) + 1);
      }
    }
  }

  // --- assemble the forecast ---
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  const perPlace: Array<Array<{ universe_id: number; p: number }>> = [];
  for (let k = 1; k <= places; k++) {
    const occTotal = [...occCounts[k]!.values()].reduce((a, b) => a + b, 0);
    const rows: Array<{ id: number; p: number }> = [];
    const seen = new Set<number>();
    for (const c of cand) {
      const pDiff = counts.get(c.id)![k]! / totalSamples;
      const pOcc = occTotal > 0 ? (occCounts[k]!.get(c.id) ?? 0) / occTotal : 0;
      const p = (1 - OCC_W) * pDiff + OCC_W * pOcc;
      if (p > 0) {
        rows.push({ id: c.id, p });
        seen.add(c.id);
      }
    }
    // games that held this place recently but are not diffusion candidates
    // (e.g. they left the chart) can still enter via the occupancy share
    if (OCC_W > 0) {
      for (const [id, cnt] of occCounts[k]!) {
        if (seen.has(id)) continue;
        const p = OCC_W * (cnt / occTotal);
        if (p > 0) rows.push({ id, p });
      }
    }
    // sharpen: raise to a power and renormalise (1 = off)
    let normRows = rows;
    if (SHARPEN !== 1) {
      const powered = rows.map((r) => ({ id: r.id, p: Math.pow(r.p, SHARPEN) }));
      const s = powered.reduce((a, r) => a + r.p, 0);
      normRows = s > 0 ? powered.map((r) => ({ id: r.id, p: r.p / s })) : powered;
    }
    const kept = normRows.filter((r) => r.p > MIN_P);
    kept.sort((a, b) => b.p - a.p);
    const top = kept.slice(0, MAX_ROWS_PER_PLACE);
    const sum = top.reduce((a, r) => a + r.p, 0);
    const scale = sum > 0 ? (1 - UNNAMED) / sum : 0;
    const out: Array<{ universe_id: number; p: number }> = [];
    for (const r of top) {
      const p = +(r.p * scale).toFixed(6);
      probabilities.push({ universe_id: r.id, place: k, p });
      out.push({ universe_id: r.id, p });
    }
    perPlace.push(out);
  }
  // top10: greedy by highest probability per place, distinct games
  const used = new Set<number>();
  const top10: number[] = [];
  for (let k = 0; k < places; k++) {
    const arr = perPlace[k]!.slice().sort((a, b) => b.p - a.p);
    const pickRow = arr.find((r) => !used.has(r.universe_id));
    if (pickRow !== undefined) {
      top10.push(pickRow.universe_id);
      used.add(pickRow.universe_id);
    }
  }
  for (const c of cand) {
    if (top10.length >= places) break;
    if (!used.has(c.id)) {
      top10.push(c.id);
      used.add(c.id);
    }
  }

  writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
}

main();
