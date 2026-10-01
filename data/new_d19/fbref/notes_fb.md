# Notes: evaluator and design of the feedback-only arrival (FB, 2026-09-29)

## Setting

Only the head stores a reference (plan family `shaped` of
`src/certification/certify_plan.py`: exact pieces of a_r on [0, T_xi), every
unit plans the same accelerations with the parking gaps, so the planned
clearances equal d_s and every schedule vanishes). Every follower stores
constants only and runs the COAST, S1, S2 laws on the errors
e_i = c_i - d_{s,i}, eps_i = v_{i-1} - v_i from takeover, with the received
command of its predecessor as feedforward; the head runs S1/S2 on its
reference errors with the feedforward a_r and brakes at a_b from T_xi.
Messages carry the applied command and the matching flag. At takeover
e_1(0) = 0 and e_i(0) = c_i(0) - d_{s,i} (the closures, 43.5 m in the
benchmark). The evaluator key `e0` holds these values.

## Evaluator (`src/certification/certify_fb.py`)

`certify_headref.py` of 2026-09-28 (unchanged, kept for the schedule
version) with one refinement ("FB refinement"):

* Earliest S2 switch instants. For a braking S1 stage (C2), unit k >= 2
  activates when it processes MATCHED_{k-1}, raised at t_sw_{k-1}, so
  t_act_k >= t_sw_{k-1} (the head activates at 0); during its S1 stage
  eps_k(t) = eps_k(0) + alpha_k (t - t_act_k) + int_0^t w_k with
  int_0^t |w_k| <= Lambda^(1)_k on [0, tswb_k] (the ledger; the head has
  w_1 = 0), and the switch needs eps_k >= -eps_det at a sample, so
      t_sw_k >= tlow_k := tlow_{k-1} + (|eps_k(0)| - eps_det - Lambda^(1)_k)^+ / alpha_k,
  with tlow_0 := 0 before the head. For t < tlow_k every nested value of the
  feedback of layer k in the command of a unit behind it is a COAST or S1
  value (0 or -alpha_k), which the S1 term of the envelopes covers, so the
  S2 terms of layer k (free response over the switch box, forced part
  gamma_k Lambda^(2)_k, planned pulses) are dropped on every cell whose end
  lies strictly before tlow_k (float evaluator: cell end >= tlow_k keeps the
  terms; interval evaluator `interval_fb.py`: cell end >= tlow_k.lo keeps
  them).
* Why it is needed: without it the upper envelopes of (C7a) place the S2
  response of the whole closure (lambda^2 e_i(0) per pair, about 0.27 m/s^2
  at lambda = 0.065) at t = 0, where the planned part of a unit behind the
  head still contains the zero command held before the first delivery; the
  upper authority test then fails by about 0.6 m/s^2 for every reference.
* The Case B regression (R3, degenerate configuration of the frozen paper)
  uses the symmetric box (its followers take over slower), where no lower
  bound is formed; R1-R6 pass unchanged.

## Corrections after the first validation (2026-09-29)

The validation suite (first run) found two weaknesses of the evaluator;
both are fixed in `certify_fb.py` and `interval_fb.py`, the theory text
states them, and every record was recomputed afterwards.

* Zero command before the first delivery. The planned part P^+-_i of the
  authority test used the window [t - (i-1) dbar, t] with a_r = 0 before
  takeover, and the envelope F^+_i could at the same time contain the S2
  closure term of a pair ahead (once t >= t_low_k). Both cannot occur
  together: a nested command whose planned part is the zero held before the
  first delivery is zero as a whole (no unit of the chain has processed a
  message of the head, so every layer is in COAST; under FIFO a unit that
  holds a packet sent at theta has processed no flag sent later). The tests
  (C7a), (C7b) now use the window truncated at 0, and the reported
  envelopes (bounds u_max/u_min_before_T_xi, curves 'up'/'lo') add the zero
  command on the cells whose window reaches before takeover (the tests are
  kept as curves 'up_test'/'lo_test'). Effect: the age boundary of the
  selected design rises from 24 ms ((C7a) at 25 ms, a 10 ms cell-edge effect
  of the switch lower bound) to 25 ms ((C8b) at 26 ms); the certificate at
  20 ms is unchanged.
* Detection threshold. A sampled takeover state with |eps_1(0)| = eps_det
  exactly (head speed 10.4650 m/s) was certified with the head switching at
  its first sample, while the simulator, computing 10.45 - 10.465 in double
  precision, saw 5.7e-17 m/s more and ran one S1 sample at -alpha, 0.348
  m/s^2 below the certified lower command bound for 1 ms. The closed loop is
  discontinuous at the threshold, so the sign indicators of the envelopes
  are now non-strict (varsigma^+-_k = 1 if +-eps_k(0) + Lambda^(1)_k >=
  eps_det): the envelopes cover the S1 stage at the threshold itself. The
  benchmark has no threshold case, so its certificate is unchanged.
* Assumption 1 check: `prem_R` also tests -a_b <= a_r(T_xi^-) <= 0.

## Design (`run_fb_design.py`, record `design_fb.json`)

Rule: a_r = min(-a_cr, cap_k) on the planning periods [k T_p, (k+1) T_p),
cap_k the largest rate on the grid q_a with cap_k <= U - mu - F^+_i(t) on
every cell whose authority window [t - (i-1) dbar, t + cell] meets the
period, for every unit i; the periods before (max_i tswb_i + hold pad) take
their minimum (no reference jump before the switches, (C0a)); a_cr on the
grid q_c makes v_r(T_xi) equal to the handoff speed; T_xi is the entry
deadline plus (n-1) dbar plus T_pad on the planning grid. The envelopes
contain the pulses of the reference's own jumps, so the rule is iterated;
from the third iteration on it is monotone (rates only decrease, T_xi only
increases, both on finite grids), and it stops when T_xi repeats and every
period's rate repeats up to ten steps q_c. Without monotonicity the rule
alternated between two iterates (four periods flipping by one step q_a and
the crawl rate by 4e-6 m/s^2).

Fixed choices: v_in = 10.45 m/s, T_p = 0.25 s, mu = 0.02 m/s^2, hold pad
0.5 s, handoff speed 1 m/s, T_pad = 2 s, q_a = 1e-3 m/s^2, q_c = 1e-6 m/s^2,
design age bound 20 ms; parking gaps 6.5 m for the followers (head slot
6 m). Selection: the smallest completion bound among the gains of
{0.05, 0.055, 0.06, 0.065, 0.07} s^-1 certified at 20 ms.

Scratch history (before the production script): the admission condition
(C9)(b) of the tail pair failed with 5 m gaps (by 0.95 m at 0.06 s^-1, by
0.22 m even at 0.045 s^-1) and passed with 6 m and 6.5 m gaps; the
sequential gaps of the selected design are (3.48, 4.04, 4.96, 6.34) m at the
reserve 0.15 m.
