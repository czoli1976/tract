#!/bin/bash
cd "$(dirname "$0")"
declare -A CLIPS=( [debugging]=Debugging_1920x1080_30fps_8bit_420.y4m [wikipedia]=Wikipedia_1920x1080p30.y4m [slides1]=Slides1_1920x1080_30fps_8bit_420.y4m [spreadsheet]=Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m )
# full CBR matrix
for short in debugging wikipedia slides1 spreadsheet; do
  for tbr in 750 1500 2500 4000; do
    for enc in base v2; do
      ./svtenc_$enc -i clips/${CLIPS[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 2 --tbr $tbr --enable-stat-report 1 -b results/cbrm_${short}_${tbr}_${enc}.ivf > results/cbrm_${short}_${tbr}_${enc}.log 2>&1
    done
  done
  echo "cbr done $short"
done
# conformance: v2 CQP points, recon vs aomdec
pass=0; fail=0
for short in debugging wikipedia slides1 spreadsheet; do
  for qp in 23 31 43 55; do
    ./svtenc_v2 -i clips/${CLIPS[$short]} -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp $qp -b conf.ivf -o recon.yuv > /dev/null 2>&1
    if ! cmp -s conf.ivf results/${short}_q${qp}_v2.ivf; then echo "BITSTREAM MISMATCH (recon tap changed output) $short q$qp"; fi
    aomdec --rawvideo -o dec.yuv conf.ivf > /dev/null 2>&1
    if cmp -s recon.yuv dec.yuv; then pass=$((pass+1)); else fail=$((fail+1)); echo "CONFORMANCE FAIL $short q$qp"; fi
    rm -f conf.ivf recon.yuv dec.yuv
  done
  echo "conformance done $short (pass=$pass fail=$fail)"
done
echo "CQP conformance: pass=$pass fail=$fail"
# decode-clean pass over CBR + special streams with dav1d and aomdec
bad=0
for f in results/cbrm_*_v2.ivf results/frozen_v2.ivf results/kf_v2.ivf results/scm0_v2.ivf; do
  dav1d -i "$f" -o /dev/null > /dev/null 2>&1 || { echo "DAV1D DECODE FAIL $f"; bad=$((bad+1)); }
  aomdec --rawvideo -o /dev/null "$f" > /dev/null 2>&1 || { echo "AOMDEC DECODE FAIL $f"; bad=$((bad+1)); }
done
echo "decode-clean failures: $bad"
echo FINAL_DONE
