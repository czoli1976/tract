#!/bin/bash
set -e
cd "$(dirname "$0")/svt"
SRC=../clips/Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m
ENC=Bin/Release/SvtAv1EncApp
run_variant() {  # $1=name
  cmake --build build -j4 > /dev/null 2>&1 || { echo "BUILD FAIL $1"; git checkout -- Source; exit 1; }
  for qp in 23 31 43 55; do
    $ENC -i $SRC -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp $qp --enable-stat-report 1 -b ../results/sp_$1_q${qp}.ivf > ../results/sp_$1_q${qp}.log 2>&1
  done
  echo "done $1"
}
# E1: residual floor 4 -> 16
sed -i 's/#define RTC_INTER_PALETTE_RES_FLOOR 4/#define RTC_INTER_PALETTE_RES_FLOOR 16/' Source/Lib/Codec/mode_decision.c
run_variant e1
git checkout -- Source/Lib/Codec/mode_decision.c
# E2: inter palette only >= 16x16
python3 - <<'PYEOF'
p = "Source/Lib/Codec/mode_decision.c"
s = open(p).read()
old = "    if (pcs->slice_type != I_SLICE) {\n        uint32_t best_me = (uint32_t)~0;"
new = "    if (pcs->slice_type != I_SLICE) {\n        if (ctx->blk_geom->bwidth < 16 || ctx->blk_geom->bheight < 16) {\n            return;\n        }\n        uint32_t best_me = (uint32_t)~0;"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PYEOF
run_variant e2
git checkout -- Source/Lib/Codec/mode_decision.c
# E3: inter palette level 7 -> 8 (FTR block only)
python3 - <<'PYEOF'
p = "Source/Lib/Codec/enc_mode_config.c"
s = open(p).read()
old = """            pcs->palette_level = is_islice ? 5 : 7;
        } else if (enc_mode <= ENC_M8) {
            pcs->palette_level = 7;"""
new = """            pcs->palette_level = is_islice ? 5 : 8;
        } else if (enc_mode <= ENC_M8) {
            pcs->palette_level = is_islice ? 7 : 8;"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
PYEOF
run_variant e3
git checkout -- Source/Lib/Codec/enc_mode_config.c
# restore clean v2 lib and canary-check against the published matrix
cmake --build build -j4 > /dev/null 2>&1
$ENC -i $SRC -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 43 -b ../results/canary_v2_q43.ivf > /dev/null 2>&1
if cmp -s ../results/canary_v2_q43.ivf ../results/spreadsheet_q43_v2.ivf; then echo "CANARY OK: clean v2 restored, matches published matrix"; else echo "CANARY MISMATCH"; fi
echo EXPERIMENTS_DONE
