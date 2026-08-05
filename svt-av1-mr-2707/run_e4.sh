#!/bin/bash
set -e
cd "$(dirname "$0")/svt"
ENC=Bin/Release/SvtAv1EncApp
python3 - <<'PYEOF'
p = "Source/Lib/Codec/palette.c"
s = open(p).read()
old = "    if (colors <= 1 || colors > 64) {\n        return;\n    }"
new = """    if (colors <= 1 || colors > 64) {
        return;
    }
#if FTR_RTC_INTER_PALETTE
    if (pcs->slice_type != I_SLICE && (ctx->blk_geom->bwidth < 16 || ctx->blk_geom->bheight < 16) && colors > 4) {
        return;
    }
#endif"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PYEOF
cmake --build build -j4 > /dev/null 2>&1 || { echo BUILD_FAIL; git checkout -- Source; exit 1; }
declare -A C=( [debugging]=Debugging_1920x1080_30fps_8bit_420.y4m [wikipedia]=Wikipedia_1920x1080p30.y4m [slides1]=Slides1_1920x1080_30fps_8bit_420.y4m [spreadsheet]=Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m )
for short in debugging wikipedia slides1 spreadsheet; do
  for qp in 23 31 43 55; do
    $ENC -i ../clips/${C[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp $qp --enable-stat-report 1 -b ../results/e4_${short}_q${qp}.ivf > ../results/e4_${short}_q${qp}.log 2>&1
  done
  for tbr in 750 1500 2500 4000; do
    $ENC -i ../clips/${C[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 2 --tbr $tbr --enable-stat-report 1 -b ../results/e4cbr_${short}_${tbr}.ivf > ../results/e4cbr_${short}_${tbr}.log 2>&1
  done
  echo "e4 done $short"
done
$ENC -i ../clips/wiki_frozen60.y4m -n 60 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b ../results/e4_frozen.ivf > /dev/null 2>&1
cmp -s ../results/e4_frozen.ivf ../results/frozen_base.ivf && echo "E4 FROZEN OK" || echo "E4 FROZEN MISMATCH"
git checkout -- Source/Lib/Codec/palette.c
cmake --build build -j4 > /dev/null 2>&1
$ENC -i ../clips/${C[spreadsheet]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 43 -b ../results/canary3.ivf > /dev/null 2>&1
cmp -s ../results/canary3.ivf ../results/spreadsheet_q43_v2.ivf && echo "CANARY OK" || echo "CANARY MISMATCH"
echo E4_DONE
