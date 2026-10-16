# Plan 2: monotonicity-only losses, post-hoc frontier

Scope (user): L may depend only on "rho increases with depth" in each sample. No reference profile, no zero_mean / mean_density, no stratification margin. The one exception is a margin at float32 ulp level (m ~ 1e-4), reported separately as a numerical sensitivity, not as physics. Post-hoc only. Guided sampling needs user approval.

## Setup (the worker's design, kept)
x0 = the 8 states of `test_new_sampler_clip/none`. Solve min_x ||x - x0||^2 + lam * L(x) over ocean cells in normalised space, then BorderZero, then evaluate. g = rho[k+1] - rho[k] in float32 torch (as in guidance), w = lower-cell volume, E in float64 as in the eval. Candidates C0-C5 as proposed. Fixed details:
- C2 tau in {1e-4, 1e-3}.
- C3 tau in {1e-3, 1e-4} with continuation, because sigmoid(-g/tau) has a vanishing gradient for g << -tau, so large inversions get no push. Start from tau = 1e-2 and divide by 3 every K iterations. Report C3 with and without continuation: this matters for transfer (below).
- C5 is allowed: a mean of monotone columns is monotone, so it is implied, not imposed. It is tested alone and added to C1 (C1 + C5).

## Fairness of the lam sweep
- Never compare at equal lam. Losses differ by orders of magnitude, so comparisons are made at matched distance.
- Scale: lam = lam_tilde / L(x0), with L(x0) averaged over the 8 states. Sweep lam_tilde over 1e-3 to 1e4 in half-decades. Then add bisection points until every loss brackets each target distance d* in {0.02, 0.044 (data-data), 0.08, 0.10}, using the same distance definition as FINDINGS.
- Interpolate E(d) per state on log(lam), then compare losses paired on the same 8 x0 (mean and min/max over states).

## Optimizer budget
- LBFGS with strong-Wolfe line search, float32 (as guidance), 300 iterations, warm-started along the lam sweep (low to high).
- Convergence check: at d* = 0.08, rerun 3x the iterations for every loss. If E or the distance moves by more than 5 % (relative), the budget is too small for that loss: raise it for all losses and flag it.
- Same budget for every candidate. Wall time is reported but not used for selection.

## Known trap: penalties stop just below 0
With a smooth penalty such as relu(-g)^2 (C0, C1), the gradient vanishes at g = 0, so solutions sit at g = -O(1/lam) and still count in the strict E.
- Always report the final |g| bins: [<1e-5, 1e-5 to 1.2e-4, 1.2e-4 to 1e-3, 1e-3 to 1e-2, >=1e-2].
- Report E and E_1e-4 = E restricted to g <= -1e-4.
- If more than half of the residual E of the best loss is |g| < 1.2e-4, run the flagged variant g -> g - 1e-4 on that loss only.

## Acceptance band and ranking (revised after user input, replaces the "2x better than C0" rule)
The user's band: a density-only loss is acceptable if its E lands between guided mean_density (sqrt eta 0.1: 2.42 %) and NONE + BorderZero (9.28 %). Going far below the data (E < 0.36 %) is a WARNING: it suggests a hidden margin or over-stratification (e.g. C2: softplus has gradient 0.5*w at g = 0). It is not a win.
- Matched-E points: report the distance (from BorderZero(x0)) each loss needs to reach E = 5 % and E = 2.5 %, plus the E reached at d = 0.01 and 0.02. Drop E = 0.36 / 0.1 as targets. Keep them only as a flag if they are reached.
- C0's plateau at 1.05 % (lam 1e3 to 1e5) is an optimizer stall. C0 is compared at the matched-E points only.
Ranking inside the band, in this order:
1. Gate, behaviour on data: at the lam that gives E = 2.5 % on NONE, applied to 16 DATA states, the distance moved must be at most 0.005 and E must not increase. Otherwise reject.
2. Gate, no hidden margin / profile freedom: cells already stable in x0 (g_old >= 0) must stay put. Report mean(g_new - g_old) over them, compared with C0 and C7. Check the |g| histogram for a pile-up at g ~ tau > 0. Level-mean variance at matched E must not fall below the data's 1.15e-4 or more than 10 % below C0's. Otherwise reject.
3. Transfer (transfer.py, t = 100/300/600): gradient cosine vs the clean gradient, active fraction on DATA x0_hat (should stay near data E, not ~100 %), and L(data) q0.5/q0.9 stable and > 0 for eps. Better transfer ranks higher.
4. Distance at matched E (smaller is better). Roughness must stay within +10 % of C0.

## Transfer to guidance (post-hoc != guided)
Guidance applies grad L to x0_hat(t), which is blurred and noisy early. The primal-dual sampler needs eps = L(data) quantile. Cheap proxies, no sampling:
- (a) Gradient on degraded inputs: evaluate grad L on x0 + noise and on smoothed x0 at 3 levels. Measure cosine similarity with grad L(x0) and the fraction of cells with a nonzero gradient. A gradient that vanishes or becomes erratic transfers badly. This is C3 without continuation, and possibly C2 with tiny tau.
- (b) eps sanity: L(data) per state, q0.5 and q0.9. It must be well above 0 and stable. C3 has the advantage that eps reads directly as a volume fraction (data E 0.36 %).

## Selection for guided sampling (approval needed)
- At most 2 candidates that pass gates 1-2, ranked by 3 then 4.
- Prefer margin-free losses (C7, C6, C1 + C5) over C2 unless C2 wins 3 and 4 and passes gate 2.
- On a tie, pick the smallest change from NGD (C7 = volume-weighted one-sided Huber).
- If only C0 passes, the answer is "NGD is fine; volume weighting brings no improvement", and no guided run is needed.
- Guided run spec for approval: sqrt mode, eps = L(data) q0.5, eta chosen so guided E lands in the band (target 2.5-5 %), 2 seeds x 8 samples, 1000 steps, against NGD sqrt 0.3 and mean_density sqrt 0.1.
