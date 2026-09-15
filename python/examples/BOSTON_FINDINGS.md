# Boston (45 orbitals, 23α/23β): SQD-sampling vs SBD-native selected-CI

Target best classical energy: **-299.521 Ha**. Two approaches tried on the
same system (`atom_1752_boston`, `count_dict_1752.json` — 1,000,000 unique
bitstrings, every count=1, no real hardware weighting, missing the near-HF
determinant entirely).

## Approach 1: `run_sqd_sbd.py` (qiskit-addon-sqd sampling loop)

Repeatedly samples/subsamples the bitstring pool, runs configuration recovery,
diagonalizes each batch with SBD. Subspace growth is driven by
`--samples_per_batch` / `--max_dim` and the sampled pool, not by SBD itself.

| run | config | outcome |
|---|---|---|
| `run1` | no `--include_hf`, `--max_dim 20000`, `samples_per_batch=100000` | 11 iterations to **-299.2139831546**, still slowly improving (~0.004 Ha/iter), killed by a 48-min wall-clock timeout — never converged or crashed |
| `run3b` | `--include_hf`, `--max_dim 40000`, `--sbd_max_nb 6`, `--max_pool_size 100000` | iteration 1 lands exactly on the HF diagonal energy (-299.1721, confirms the hand-calculated lower bound), only 4 iterations to **-299.1749705518** before GPU OOM at iteration 5 (`dim` had grown to `40000² = 1.6e9`) |

**Findings:**
- `--include_hf` reliably guarantees a good, provable starting point immediately
  (skips ~2 wasted iterations stuck near -294.25 while configuration recovery
  finds its own way there) — but in these runs it never got the chance to pull
  ahead, since the larger `--max_dim` it was paired with hit GPU OOM first.
- The two runs are **not a clean A/B**: different `--max_dim`, different
  stopping cause (timeout vs OOM). `run1`'s number is nominally higher only
  because it survived longer at a smaller `--max_dim`.
- Per-iteration progress in both cases was small and non-monotonic in rate
  (~0.0006-0.013 Ha/iteration) — the sampled pool's uniform count=1 weighting
  (no real hardware signal) appears to be a real ceiling on what configuration
  recovery can find, independent of `--max_dim`.
- Cross-iteration GPU memory growth at *fixed* `dim` was also observed and is
  unresolved (see `[[boston-sqd-overnight-session-2026-09-14]]` in memory) —
  a separate, still-open issue with the Thrust backend across repeated
  `tpb_diag()` calls in one long-running process.

## Approach 2: `run_sbd_selected_ci.py` (SBD-native carryover expansion, this branch)

No `qiskit_addon_sqd` sampling loop at all. Seeds from the counts pool once
(frequency-weighted via `qiskit_addon_sqd.counts`/`subsampling` utilities,
reused only for parsing), then lets SBD's own `--sbd_carryover_type`
single-excitation expansion (`carryover_type=2`/`3`) grow the subspace every
iteration from the wavefunction SBD already computed — no re-sampling.

| carryover type | `--include_hf` | `--max_dim` | `--sbd_carryover_threshold` | best energy | stopped because |
|---|---|---|---|---|---|
| 2 | no | 5000 | 1e-3 | -299.1769651354 | hit `--max_iterations 5`, still improving (not converged) |
| 2 | yes | 5000 | 1e-3 | -299.4451964009 | converged (iter 4) |
| 2 | yes | 15000 | 1e-3 | -299.4520549989 | converged (iter 4) |
| 3 | yes | 5000 | 1e-3 | -299.4592657531 | converged (iter 3) — expansion closed naturally at 2329x2329, *never even reached* the 5000 cap |
| 3 | no | 5000 | 1e-3 | -299.4668763840 | converged (iter 7, given 12 iterations) — closed naturally at 3603x4019, also under the 5000 cap |
| 3 | no | 15000 | 1e-3 | -299.4789854650 | iter 6 of 6, still growing when the run ended (not converged) |
| 3 | no | 15000 | 1e-4 | **-299.4860424776** | converged (iter 9, 659s) |

**Findings:**
- `carryover_type=3` (dominant-amplitude + singles) consistently beats
  `carryover_type=2` (marginal-probability + singles) at the same `--max_dim`:
  -299.459/-299.467 vs -299.445 at max_dim=5000, both with plenty of headroom
  left under the cap. Type=3 finds a *smaller*, self-closing subspace that is
  nonetheless a *better* energy — it is not simply exploring more determinants.
- `--include_hf` helps type=2 substantially (its marginal-probability filter
  can otherwise drop the HF determinant before the singles expansion even
  runs: -299.177 without it vs -299.445 with it, same max_dim=5000) but is
  mildly detrimental for type=3: without `--include_hf`, given enough
  iterations, type=3 converges to a *better* energy (-299.4668763840) at a
  *larger* natural subspace (3603x4019) than with `--include_hf`
  (-299.4592657531, closed early at 2329x2329) — forcing the HF determinant
  in seems to bias the expansion toward converging prematurely at a smaller
  subspace, rather than helping.
- Lowering `--sbd_carryover_threshold` (1e-3 -> 1e-4, both type=3/no-HF/
  max_dim=15000) bought only ~0.007 Ha over the max_dim=15000/threshold=1e-3
  run (-299.4789854650, itself not yet converged when that run ended) —
  once `--max_dim=15000` starts truncating, the expanded candidate set
  reaches an exact fixed point (`17868 x 18312`, unchanged run over run) and
  the energy stops moving to the 10th decimal. `--max_dim` itself, not the
  threshold, is the binding constraint at 15000.
- No OOM, no timeout, no crash in any of these runs — SBD's own expansion
  mechanism produces a bounded, well-behaved subspace at every step, unlike
  the sampling loop's pool-driven growth.

## Bottom line

| | best energy | gap to -299.521 |
|---|---|---|
| `run_sqd_sbd.py` (sampling loop) | -299.1749705518 (OOM-limited) / -299.2139831546 (timeout-limited, unconverged) | ~0.31-0.35 Ha |
| `run_sbd_selected_ci.py` (SBD-native, type=3) | **-299.4860424776** (converged) | **~0.035 Ha** |

SBD's own carryover expansion, with no sampling and no `qiskit_addon_sqd` loop
at all, converges cleanly to a result an order of magnitude closer to the
target than the sampling approach managed overnight — and does so without
hitting any of the sampling loop's memory/timeout failure modes.

**Open next step:** try `carryover_type=3` with `--include_hf` *and* a larger
`--max_dim` together, now that `--max_dim=15000` has been identified as the
binding constraint at the current best.
