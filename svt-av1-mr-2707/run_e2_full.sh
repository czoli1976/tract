#!/bin/bash
set -e
cd "$(dirname "$0")/svt"
ENC=Bin/Release/SvtAv1EncApp
python3 - <<'PYEOF'
p = "Source/Lib/Codec/mode_decision.c"
s = open(p).read()
old = "    if (pcs->slice_type != I_SLICE) {\n        uint32_t best_me = (uint32_t)~0;"
new = "    if (pcs->slice_type != I_SLICE) {\n        if (ctx->blk_geom->bwidth < 16 || ctx->blk_geom->bheight < 16) {\n            return;\n        }\n        uint32_t best_me = (uint32_t)~0;"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PYEOF
cmake --build build -j4 > /dev/null 2>&1 || { echo BUILD_FAIL; git checkout -- Source; exit 1; }
declare -A CLIPS=( [debugging]=Debugging_1920x1080_30fps_8bit_420.y4m [wikipedia]=Wikipedia_1920x1080p30.y4m [slides1]=Slides1_1920x1080_30fps_8bit_420.y4m )
for short in debugging wikipedia slides1; do
  for qp in 23 31 43 55; do
    $ENC -i ../clips/${CLIPS[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp $qp --enable-stat-report 1 -b ../results/sp2_${short}_q${qp}.ivf > ../results/sp2_${short}_q${qp}.log 2>&1
  done
  echo "e2 cqp done $short"
done
declare -A CLIPS2=( [slides1]=Slides1_1920x1080_30fps_8bit_420.y4m [spreadsheet]=Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m [debugging]=Debugging_1920x1080_30fps_8bit_420.y4m [wikipedia]=Wikipedia_1920x1080p30.y4m )
for short in slides1 spreadsheet debugging wikipedia; do
  for tbr in 750 1500 2500 4000; do
    $ENC -i ../clips/${CLIPS2[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 2 --tbr $tbr --enable-stat-report 1 -b ../results/sp2cbr_${short}_${tbr}.ivf > ../results/sp2cbr_${short}_${tbr}.log 2>&1
  done
  echo "e2 cbr done $short"
done
# frozen-screen byte-identity must still hold for E2
$ENC -i ../clips/wiki_frozen60.y4m -n 60 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b ../results/sp2_frozen.ivf > /dev/null 2>&1
cmp -s ../results/sp2_frozen.ivf ../results/frozen_base.ivf && echo "E2 FROZEN OK (byte-identical to base)" || echo "E2 FROZEN MISMATCH"
git checkout -- Source/Lib/Codec/mode_decision.c
cmake --build build -j4 > /dev/null 2>&1
$ENC -i ../clips/${CLIPS2[spreadsheet]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 43 -b ../results/canary2.ivf > /dev/null 2>&1
cmp -s ../results/canary2.ivf ../results/spreadsheet_q43_v2.ivf && echo "CANARY OK" || echo "CANARY MISMATCH"
echo E2_FULL_DONE
