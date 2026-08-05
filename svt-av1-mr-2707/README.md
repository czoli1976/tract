# SVT-AV1 MR 2707 v2 — deliverables

Rework of https://gitlab.com/AOMediaCodec/SVT-AV1/-/merge_requests/2707
("RTC: palette on inter-frame intra blocks for screen content") addressing
Mo_Amine's review. Not tract code — this branch is a delivery vehicle only;
do not merge into main.

## Contents

- `0001-RTC-palette-on-inter-frame-intra-blocks-for-screen-c.patch` — the v2
  commit, based on SVT-AV1 master `13438c1`. 3 files, +60 lines. Changes vs v1:
  frame-idle gate moved from Picture Decision (pre-ME, read a pool-recycled
  previous-picture `norm_me_dist`) to `svt_aom_sig_deriv_mode_decision_config_rtc`
  (post-ME, reads this frame's value; idle frames drop palette and re-derive
  `allow_screen_content_tools`, signaling exactly as baseline); top-8-coverage
  gate removed; residual-floor skip kept with the `eval_intrabc`/`palette_hint`
  comment; dead `sc_class1` hunk gone with the rebase; per-branch level
  assignment resolves the `<= ENC_M8` scope nit.
- `mr2707_v2_local_results.md` — full local verification on the AOM b2_scc
  originals (all four clips, full 130 frames): byte-level correctness checks,
  CQP BD-rates, CBR spot check, timing.
- `run_matrix.sh` / `run_extra.sh` / `analyze.py` — the reproducible harness
  (expects `clips/` with the four b2_scc y4m files and `svtenc_base` /
  `svtenc_v2` binaries alongside).

## To update the MR

```
git fetch https://gitlab.com/AOMediaCodec/SVT-AV1.git master
git checkout -B ftr-rtc-inter-palette FETCH_HEAD
git am 0001-RTC-palette-on-inter-frame-intra-blocks-for-screen-c.patch
git push -f <your-fork-remote> ftr-rtc-inter-palette
```

The commit carries a `Co-Authored-By: Claude` trailer; amend it away if you
prefer sole authorship on the MR.

## Remaining before posting

- Decoder conformance (`aomdec`/`dav1d`) — not runnable in the session container.
- Optionally re-run `run_matrix.sh` on a quiet machine for the published table.
- Post the refreshed numbers with operating points: b2_scc originals, 130 frames,
  preset 8, `--rtc 1 --scm 1 --lp 1 --rc 0 --cqp {23,31,43,55}`;
  CBR `--rc 2 --tbr {1000,2500}`.
