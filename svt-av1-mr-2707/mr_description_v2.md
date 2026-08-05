# MR 2707 v2 — ready-to-paste description

Replace the MR description with the text below after force-pushing the branch.
Drag-and-drop `g1_bdrate.png`, `g2_rd.png`, `g3_cost.png`, `g4_dbgain.png` into
the editor at the four marked spots (GitLab inserts the `![...]` links for you).

---

## Summary

SVT's RTC path runs AV1 palette only on I-slices. libaom-RT (`force_palette_test`)
also evaluates palette on intra-coded blocks **inside inter frames** for screen
content — exactly where the static text/UI of a screen-share stream lives. This
MR closes that gap at presets **M7–M8**, gated so the cost tracks screen activity
and only fires where palette actually pays off.

## What it does (v2 — reworked after review)

- **Picture decision** (`svt_aom_sig_deriv_multi_processes_rtc`, sc_class5): inter
  frames also receive a palette level on detected screen content at M7–M8, set
  per preset branch so `allow_screen_content_tools` is derived coherently at its
  existing site. I-slice palette is unchanged; M9+ unchanged (SC tools off there).
- **Frame-idle gate** — moved to `svt_aom_sig_deriv_mode_decision_config_rtc`,
  post-ME, where **this frame's** `norm_me_dist` exists (at picture decision it
  does not — v1 read a pool-recycled previous-picture value). A frame with zero
  average ME distortion drops palette and re-derives the header flag, so an idle
  inter frame signals **exactly as baseline** (verified byte-identical on a
  frozen-screen clip and on Wikipedia's 12-frame static intro).
- **Per-block residual-floor skip** (`inject_palette_candidates`) — skip the
  palette search on inter blocks where inter prediction is essentially perfect
  (per-pixel ME residual ≤ 4); palette can't beat a ~0-bit zero-MV skip there.
- The v1 top-8-coverage gate is **dropped** (fired on 0.7% of calls; compiling it
  out moved BD-rate ≤0.5pt / CPU ≤0.9pt — per review measurements).

## Results

AOM `b2_scc` originals, full 130-frame clips, 1080p, preset 8,
`--rtc 1 --scm 1 --lp 1`. CQP: `--rc 0 --cqp {23,31,43,55}`.
CBR: `--rc 2 --tbr {750,1500,2500,4000}`. Y-PSNR BD-rate vs base (`13438c1`):

| clip | CQP BD-rate | CBR BD-rate |
|---|---|---|
| Debugging | −16.2% | −10.8% |
| Wikipedia | −17.1% | −12.6% |
| Slides1 | −2.0% | −0.8% |
| Spreadsheet | +1.1% | +2.6% |

- ΔPSNR at matched QP is ≥ 0 at all 16 CQP points (+0.06 to +1.66 dB).
- Encode cost is per-clip: **+2–11% (Slides1), +3–5% (Wikipedia)** on active
  screen content; ~0% on a static screen (idle gate; frozen-screen output is
  byte-identical to baseline). Camera content (`--scm 0`) byte-identical.
- Keyframes byte-identical (per IVF payload, including mid-stream KFs at
  `--keyint 60`). Deterministic across reruns and `--lp 1/4`.
- **Conformance:** all 16 CQP points — SVT recon byte-identical to `aomdec`
  decode; every CBR / frozen / multi-KF stream decodes clean through `aomdec`
  and `dav1d`.

### BD-rate per clip (CQP + CBR)
*(upload g1_bdrate.png here)*

### Rate–distortion (palette shifts the curve up / left)
*(upload g2_rd.png here)*

### Cost is demand-driven (per-frame bytes; byte-identical static intro marked)
*(upload g3_cost.png here)*

### Per-QP quality gain (dB, matched QP)
*(upload g4_dbgain.png here)*

## Known trade: Spreadsheet

Spreadsheet is the one clip that regresses (+1.1% CQP / +2.6% CBR). Investigated
rather than just reported: restricting inter-frame palette on sub-16×16 blocks
(none / ≤4 colors / exact-palette-only) flips Spreadsheet negative (best −3.0%)
but surrenders ~40% of the Debugging/Wikipedia gains — the many-color small
blocks that lose on dense spreadsheet cells are the same signature that wins
biggest on sparse text, and no cheap block-local or frame-level statistic
separates them (measured; details in review thread). Shipping without the gate
and stating the trade; if a no-regression profile is preferred, the
exact-palette-only variant (2 lines in `search_palette_luma`) gives
−10.3 / −10.3 / −2.9 / −2.6 CQP with all-negative CBR. Likeliest real fix is
content-agnostic: audit palette index-map rate estimation at MD vs actual coded
bits (follow-up).

## Competitive context vs libaom-RT

The v1 comparison (SVT preset 8 vs libaom-RT `--usage=1 --cpu-used=6
--tune-content=screen`, CQP/Y-PSNR) showed SVT+palette widening SVT's existing
lead by ~10–12pp on the text-heavy clips. Those numbers were measured with v1;
v2's larger gains on the same clips make them conservative. Will refresh on
request.
