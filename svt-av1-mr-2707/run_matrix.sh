#!/bin/bash
cd "$(dirname "$0")"
declare -A CLIPS=( [debugging]=Debugging_1920x1080_30fps_8bit_420.y4m [wikipedia]=Wikipedia_1920x1080p30.y4m [slides1]=Slides1_1920x1080_30fps_8bit_420.y4m [spreadsheet]=Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m )
for short in debugging wikipedia slides1 spreadsheet; do
  for qp in 23 31 43 55; do
    for enc in base v2; do
      ./svtenc_$enc -i clips/${CLIPS[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp $qp --enable-stat-report 1 -b results/${short}_q${qp}_${enc}.ivf > results/${short}_q${qp}_${enc}.log 2>&1
    done
  done
  echo "done $short"
done
# determinism: same cmd twice, plus lp4
./svtenc_v2 -i clips/${CLIPS[wikipedia]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b results/det_run2.ivf > /dev/null 2>&1
./svtenc_v2 -i clips/${CLIPS[wikipedia]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 4 --rc 0 --cqp 31 -b results/det_lp4.ivf > /dev/null 2>&1
# scm 0 scoping: feature must be inert
./svtenc_base -i clips/${CLIPS[wikipedia]} -n 130 --preset 8 --rtc 1 --scm 0 --lp 1 --rc 0 --cqp 31 -b results/scm0_base.ivf > /dev/null 2>&1
./svtenc_v2   -i clips/${CLIPS[wikipedia]} -n 130 --preset 8 --rtc 1 --scm 0 --lp 1 --rc 0 --cqp 31 -b results/scm0_v2.ivf   > /dev/null 2>&1
# multi-keyframe run for KF byte-identity check
./svtenc_base -i clips/${CLIPS[debugging]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 --keyint 60 -b results/kf_base.ivf > /dev/null 2>&1
./svtenc_v2   -i clips/${CLIPS[debugging]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 --keyint 60 -b results/kf_v2.ivf   > /dev/null 2>&1
echo ALL_DONE
